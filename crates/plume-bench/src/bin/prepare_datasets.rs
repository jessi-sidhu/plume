//! Prepare the Phase 1 pruning datasets under ./data (gitignored).
//!
//! - TPC-H SF1 `lineitem` and `orders`, generated with DuckDB's `tpch`
//!   extension (`CALL dbgen(sf=1)`).
//! - NYC TLC yellow taxi trips, January–June 2023, downloaded from the TLC
//!   trip-record page and merged into one table.
//!
//! Plume has no DECIMAL type, so every DECIMAL column is cast to DOUBLE on
//! export. The list of cast columns is read from DuckDB's catalog and
//! recorded in the manifest.
//!
//! Each fact table (`lineitem`, taxi trips) is written in three layouts,
//! all with the same row-group size:
//!   - random:   ORDER BY hash(rowid)
//!   - sorted:   ORDER BY l_shipdate            / tpep_pickup_datetime
//!   - compound: ORDER BY l_shipdate, l_quantity / tpep_pickup_datetime, PULocationID
//!
//! Every ORDER BY ends with `rowid` as a tiebreaker so the output is
//! byte-for-byte reproducible; it only orders rows whose sort keys are equal.
//! `orders` is the join build side, so it is written once, in dbgen order.
//!
//! Finally `data/manifest.json` records each file's row count, row-group
//! count and size (read back from the written files) plus the sources.
//! Row counts must match across layouts or the run fails.
//!
//! Run from anywhere:
//!     cargo run --release -p plume-bench --bin prepare_datasets
//! Options:
//!     --tpch-sf <f>      TPC-H scale factor (default 1)
//!     --only tpch|taxi   prepare one dataset

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, ensure, Context, Result};
use duckdb::Connection;
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde_json::{json, Value};

const ROW_GROUP_SIZE: usize = 122_880;
const TAXI_MONTHS: [&str; 6] = [
    "2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06",
];
const TAXI_URL_BASE: &str = "https://d37ci6vzurychx.cloudfront.net/trip-data";
const TAXI_SOURCE_PAGE: &str = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page";

struct Layout {
    name: &'static str,
    order_by: &'static str,
}

const LINEITEM_LAYOUTS: [Layout; 3] = [
    Layout {
        name: "random",
        order_by: "hash(rowid), rowid",
    },
    Layout {
        name: "sorted",
        order_by: "l_shipdate, rowid",
    },
    Layout {
        name: "compound",
        order_by: "l_shipdate, l_quantity, rowid",
    },
];

const TAXI_LAYOUTS: [Layout; 3] = [
    Layout {
        name: "random",
        order_by: "hash(rowid), rowid",
    },
    Layout {
        name: "sorted",
        order_by: "tpep_pickup_datetime, rowid",
    },
    Layout {
        name: "compound",
        order_by: "tpep_pickup_datetime, PULocationID, rowid",
    },
];

struct Args {
    tpch_sf: f64,
    tpch: bool,
    taxi: bool,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        tpch_sf: 1.0,
        tpch: true,
        taxi: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--tpch-sf" => {
                let v = it.next().context("--tpch-sf needs a value")?;
                args.tpch_sf = v.parse().with_context(|| format!("bad --tpch-sf {v}"))?;
            }
            "--only" => match it.next().as_deref() {
                Some("tpch") => args.taxi = false,
                Some("taxi") => args.tpch = false,
                other => bail!("--only expects tpch or taxi, got {other:?}"),
            },
            other => bail!("unknown argument {other}"),
        }
    }
    Ok(args)
}

fn workspace_root() -> Result<PathBuf> {
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .context("locate workspace root from CARGO_MANIFEST_DIR")?
        .to_path_buf())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let data = workspace_root()?.join("data");
    let tmp = data.join("tmp");
    fs::create_dir_all(&tmp)?;

    // File-backed so large sorts can spill to disk instead of exhausting RAM.
    let db_path = tmp.join("prepare.duckdb");
    remove_db(&db_path)?;
    let conn = Connection::open(&db_path).context("open duckdb")?;
    conn.execute_batch(&format!(
        "SET temp_directory = '{}'; SET preserve_insertion_order = true;",
        sql_path(&tmp.join("spill"))?
    ))?;
    let duckdb_version: String = conn.query_row("SELECT version()", [], |r| r.get(0))?;

    let mut datasets = serde_json::Map::new();
    if args.tpch {
        datasets.insert(
            "tpch".into(),
            prepare_tpch(&conn, &data, args.tpch_sf, &duckdb_version)?,
        );
    }
    if args.taxi {
        datasets.insert("taxi".into(), prepare_taxi(&conn, &data)?);
    }
    drop(conn);
    remove_db(&db_path)?;
    let _ = fs::remove_dir_all(tmp.join("spill"));

    // Merge with an existing manifest so `--only` doesn't drop the other
    // dataset's entry.
    let manifest_path = data.join("manifest.json");
    let mut all = match fs::read_to_string(&manifest_path) {
        Ok(s) => serde_json::from_str::<Value>(&s)
            .ok()
            .and_then(|v| v.get("datasets").cloned())
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default(),
        Err(_) => serde_json::Map::new(),
    };
    all.extend(datasets);
    let manifest = json!({
        "generated_by": "cargo run --release -p plume-bench --bin prepare_datasets",
        "duckdb_version": duckdb_version,
        "row_group_size": ROW_GROUP_SIZE,
        "datasets": all,
    });
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;
    println!("wrote {}", manifest_path.display());
    Ok(())
}

fn prepare_tpch(conn: &Connection, data: &Path, sf: f64, duckdb_version: &str) -> Result<Value> {
    let dir = data.join("tpch");
    fs::create_dir_all(&dir)?;
    println!("tpch: generating SF{sf} with dbgen");
    conn.execute_batch(&format!("INSTALL tpch; LOAD tpch; CALL dbgen(sf={sf});"))?;

    let mut files = serde_json::Map::new();
    let mut casts = serde_json::Map::new();

    let (select, cast) = select_list(conn, "lineitem")?;
    casts.insert("lineitem".into(), json!(cast));
    let layouts = write_layouts(
        conn,
        "lineitem",
        &select,
        &LINEITEM_LAYOUTS,
        &dir,
        "lineitem",
    )?;
    files.insert("lineitem".into(), layouts);

    let (select, cast) = select_list(conn, "orders")?;
    casts.insert("orders".into(), json!(cast));
    let orders_path = dir.join("orders.parquet");
    copy_to_parquet(
        conn,
        &format!("SELECT {select} FROM orders ORDER BY rowid"),
        &orders_path,
    )?;
    files.insert(
        "orders".into(),
        json!({ "dbgen_order": file_info(&orders_path)? }),
    );

    for t in [
        "lineitem", "orders", "customer", "part", "partsupp", "supplier", "nation", "region",
    ] {
        conn.execute_batch(&format!("DROP TABLE IF EXISTS {t};"))?;
    }

    Ok(json!({
        "source": format!("DuckDB tpch extension, CALL dbgen(sf={sf}), DuckDB {duckdb_version}"),
        "scale_factor": sf,
        "decimal_columns_cast_to_double": casts,
        "files": files,
    }))
}

fn prepare_taxi(conn: &Connection, data: &Path) -> Result<Value> {
    let raw = data.join("raw").join("taxi");
    let dir = data.join("taxi");
    fs::create_dir_all(&raw)?;
    fs::create_dir_all(&dir)?;

    let mut sources = Vec::new();
    let mut raw_paths = Vec::new();
    for month in TAXI_MONTHS {
        let name = format!("yellow_tripdata_{month}.parquet");
        let url = format!("{TAXI_URL_BASE}/{name}");
        let path = raw.join(&name);
        download(&url, &path)?;
        sources.push(json!({ "url": url, "raw": file_info(&path)? }));
        raw_paths.push(format!("'{}'", sql_path(&path)?));
    }

    // Merge the six months into one table. union_by_name lines columns up
    // by name in case a month adds, drops or reorders a column.
    println!("taxi: merging {} monthly files", raw_paths.len());
    conn.execute_batch(&format!(
        "CREATE OR REPLACE TABLE trips AS \
         SELECT * FROM read_parquet([{}], union_by_name = true);",
        raw_paths.join(", ")
    ))?;

    let (select, cast) = select_list(conn, "trips")?;
    let layouts = write_layouts(conn, "trips", &select, &TAXI_LAYOUTS, &dir, "trips")?;
    conn.execute_batch("DROP TABLE trips;")?;

    Ok(json!({
        "source": TAXI_SOURCE_PAGE,
        "months": TAXI_MONTHS,
        "inputs": sources,
        "decimal_columns_cast_to_double": cast,
        "files": { "trips": layouts },
    }))
}

/// Write `table` once per layout and check that every layout has the same
/// row count.
fn write_layouts(
    conn: &Connection,
    table: &str,
    select: &str,
    layouts: &[Layout],
    dir: &Path,
    file_stem: &str,
) -> Result<Value> {
    let mut out = serde_json::Map::new();
    let mut row_counts = Vec::new();
    for layout in layouts {
        let path = dir.join(format!("{file_stem}_{}.parquet", layout.name));
        println!("{table}: writing {} layout", layout.name);
        copy_to_parquet(
            conn,
            &format!("SELECT {select} FROM {table} ORDER BY {}", layout.order_by),
            &path,
        )?;
        let info = file_info(&path)?;
        row_counts.push(info["rows"].as_i64().unwrap_or(-1));
        let mut entry = info.as_object().cloned().unwrap_or_default();
        entry.insert("order_by".into(), json!(layout.order_by));
        out.insert(layout.name.into(), Value::Object(entry));
    }
    ensure!(
        row_counts.windows(2).all(|w| w[0] == w[1]),
        "{table}: row counts differ across layouts: {row_counts:?}"
    );
    let source_rows: i64 =
        conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))?;
    ensure!(
        row_counts[0] == source_rows,
        "{table}: wrote {} rows, source table has {source_rows}",
        row_counts[0]
    );
    Ok(Value::Object(out))
}

/// SELECT list for `table` with every DECIMAL column cast to DOUBLE.
/// Returns the list and the names of the cast columns.
fn select_list(conn: &Connection, table: &str) -> Result<(String, Vec<String>)> {
    let mut stmt = conn.prepare(
        "SELECT column_name, data_type FROM information_schema.columns \
         WHERE table_name = ? ORDER BY ordinal_position",
    )?;
    let cols: Vec<(String, String)> = stmt
        .query_map([table], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<std::result::Result<_, _>>()?;
    ensure!(!cols.is_empty(), "table {table} has no columns");
    let mut cast = Vec::new();
    let select = cols
        .iter()
        .map(|(name, ty)| {
            if ty.starts_with("DECIMAL") {
                cast.push(name.clone());
                format!("CAST(\"{name}\" AS DOUBLE) AS \"{name}\"")
            } else {
                format!("\"{name}\"")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    Ok((select, cast))
}

fn copy_to_parquet(conn: &Connection, query: &str, path: &Path) -> Result<()> {
    let _ = fs::remove_file(path);
    conn.execute_batch(&format!(
        "COPY ({query}) TO '{}' (FORMAT PARQUET, ROW_GROUP_SIZE {ROW_GROUP_SIZE});",
        sql_path(path)?
    ))
    .with_context(|| format!("COPY to {}", path.display()))
}

/// Row count, row-group count and size, read back from the written file.
fn file_info(path: &Path) -> Result<Value> {
    let reader = SerializedFileReader::new(
        File::open(path).with_context(|| format!("open {}", path.display()))?,
    )?;
    let meta = reader.metadata();
    Ok(json!({
        "path": path
            .strip_prefix(workspace_root()?)
            .unwrap_or(path)
            .display()
            .to_string(),
        "rows": meta.file_metadata().num_rows(),
        "row_groups": meta.num_row_groups(),
        "bytes": fs::metadata(path)?.len(),
    }))
}

/// Download `url` to `path` with curl unless it is already there.
fn download(url: &str, path: &Path) -> Result<()> {
    if path.exists() {
        println!("taxi: have {}", path.display());
        return Ok(());
    }
    println!("taxi: downloading {url}");
    let part = path.with_extension("parquet.part");
    let status = Command::new("curl")
        .args(["--fail", "--location", "--silent", "--show-error"])
        .args(["--retry", "3", "--output"])
        .arg(&part)
        .arg(url)
        .status()
        .context("run curl (is it installed?)")?;
    ensure!(status.success(), "curl failed for {url}");
    fs::rename(&part, path)?;
    Ok(())
}

fn sql_path(p: &Path) -> Result<String> {
    let s = p.to_str().context("non-UTF-8 path")?;
    ensure!(!s.contains('\''), "path contains a quote: {s}");
    Ok(s.to_string())
}

fn remove_db(p: &Path) -> Result<()> {
    for f in [p.to_path_buf(), p.with_extension("duckdb.wal")] {
        if f.exists() {
            fs::remove_file(&f)?;
        }
    }
    Ok(())
}
