//! Input/output adapters for CSV, JSONL/NDJSON, and Apache Parquet.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt32Type, UInt64Type};
use arrow_array::Array;
use arrow_schema::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ShapDriver {
    pub feature: String,
    pub label: String,
    pub value: String,
    pub shap_value: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoredRow {
    pub row_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub raw_score: f64,
    pub prediction: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_drivers: Option<Vec<ShapDriver>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shap_values: Option<HashMap<String, f64>>,
}

pub fn read_records(path: &Path) -> Result<Vec<HashMap<String, String>>, String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "csv" | "tsv" => read_csv(path, ext == "tsv"),
        "jsonl" | "ndjson" | "json" => read_jsonl(path),
        "parquet" | "pq" => read_parquet(path),
        _ => read_csv(path, false),
    }
}

fn read_csv(path: &Path, is_tsv: bool) -> Result<Vec<HashMap<String, String>>, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(if is_tsv { b'\t' } else { b',' })
        .from_path(path)
        .map_err(|e| format!("failed to open CSV {}: {e}", path.display()))?;
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("failed to read CSV headers: {e}"))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut out = Vec::new();
    for rec in rdr.records() {
        let rec = rec.map_err(|e| format!("CSV row parse error: {e}"))?;
        let mut map = HashMap::with_capacity(headers.len());
        for (h, v) in headers.iter().zip(rec.iter()) {
            map.insert(h.clone(), v.to_string());
        }
        out.push(map);
    }
    Ok(out)
}

fn read_jsonl(path: &Path) -> Result<Vec<HashMap<String, String>>, String> {
    let f = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("read error at line {}: {e}", line_no + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val: serde_json::Value = serde_json::from_str(trimmed)
            .map_err(|e| format!("invalid JSON at line {}: {e}", line_no + 1))?;
        if let Some(arr) = val.as_array() {
            for item in arr {
                out.push(json_obj_to_map(item)?);
            }
        } else {
            out.push(json_obj_to_map(&val)?);
        }
    }
    Ok(out)
}

pub fn json_obj_to_map(val: &serde_json::Value) -> Result<HashMap<String, String>, String> {
    let obj = val
        .as_object()
        .ok_or_else(|| "expected JSON object row".to_string())?;
    let mut map = HashMap::with_capacity(obj.len());
    for (k, v) in obj {
        let s = match v {
            serde_json::Value::Null => String::new(),
            serde_json::Value::String(st) => st.clone(),
            other => other.to_string(),
        };
        map.insert(k.clone(), s);
    }
    Ok(map)
}

fn read_parquet(path: &Path) -> Result<Vec<HashMap<String, String>>, String> {
    let f = File::open(path).map_err(|e| format!("failed to open Parquet {}: {e}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(f)
        .map_err(|e| format!("failed to read Parquet metadata: {e}"))?;
    let reader = builder
        .build()
        .map_err(|e| format!("failed to build Parquet reader: {e}"))?;
    let mut out = Vec::new();
    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("Parquet batch error: {e}"))?;
        let schema = batch.schema();
        let n_rows = batch.num_rows();
        let n_cols = batch.num_columns();
        for row_idx in 0..n_rows {
            let mut map = HashMap::with_capacity(n_cols);
            for col_idx in 0..n_cols {
                let col_name = schema.field(col_idx).name().clone();
                let arr = batch.column(col_idx);
                if arr.is_null(row_idx) {
                    map.insert(col_name, String::new());
                    continue;
                }
                let s = match arr.data_type() {
                    DataType::Float64 => arr.as_primitive::<Float64Type>().value(row_idx).to_string(),
                    DataType::Float32 => arr.as_primitive::<Float32Type>().value(row_idx).to_string(),
                    DataType::Int64 => arr.as_primitive::<Int64Type>().value(row_idx).to_string(),
                    DataType::Int32 => arr.as_primitive::<Int32Type>().value(row_idx).to_string(),
                    DataType::Int16 => arr.as_primitive::<Int16Type>().value(row_idx).to_string(),
                    DataType::Int8 => arr.as_primitive::<Int8Type>().value(row_idx).to_string(),
                    DataType::UInt64 => arr.as_primitive::<UInt64Type>().value(row_idx).to_string(),
                    DataType::UInt32 => arr.as_primitive::<UInt32Type>().value(row_idx).to_string(),
                    DataType::Boolean => arr.as_boolean().value(row_idx).to_string(),
                    DataType::Utf8 => arr.as_string::<i32>().value(row_idx).to_string(),
                    DataType::LargeUtf8 => arr.as_string::<i64>().value(row_idx).to_string(),
                    _ => String::new(),
                };
                map.insert(col_name, s);
            }
            out.push(map);
        }
    }
    Ok(out)
}

pub fn write_scored_output(path: &Path, rows: &[ScoredRow]) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed creating dir {}: {e}", parent.display()))?;
    }
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext == "csv" {
        let mut wtr = csv::Writer::from_path(path)
            .map_err(|e| format!("failed creating CSV {}: {e}", path.display()))?;
        wtr.write_record(["row_index", "id", "raw_score", "prediction", "top_drivers"])
            .map_err(|e| format!("CSV header error: {e}"))?;
        for r in rows {
            let drivers_str = r
                .top_drivers
                .as_ref()
                .map(|ds| {
                    ds.iter()
                        .map(|d| format!("{}: {} ({:+.4})", d.label, d.value, d.shap_value))
                        .collect::<Vec<_>>()
                        .join(" | ")
                })
                .unwrap_or_default();
            wtr.write_record([
                r.row_index.to_string(),
                r.id.clone().unwrap_or_default(),
                format!("{:.6}", r.raw_score),
                format!("{:.6}", r.prediction),
                drivers_str,
            ])
            .map_err(|e| format!("CSV write error: {e}"))?;
        }
        wtr.flush().map_err(|e| format!("CSV flush error: {e}"))?;
    } else {
        let f = File::create(path)
            .map_err(|e| format!("failed creating JSONL {}: {e}", path.display()))?;
        let mut bw = BufWriter::new(f);
        for r in rows {
            let line = serde_json::to_string(r).map_err(|e| format!("JSON encode error: {e}"))?;
            writeln!(bw, "{line}").map_err(|e| format!("write error: {e}"))?;
        }
        bw.flush().map_err(|e| format!("flush error: {e}"))?;
    }
    Ok(())
}

pub fn load_label_map(path: Option<&Path>) -> Result<HashMap<String, String>, String> {
    let Some(p) = path else {
        return Ok(HashMap::new());
    };
    let text = fs::read_to_string(p)
        .map_err(|e| format!("failed reading labels file {}: {e}", p.display()))?;
    let val: toml::Value =
        toml::from_str(&text).map_err(|e| format!("invalid labels TOML: {e}"))?;
    let mut map = HashMap::new();
    let table = val
        .get("labels")
        .and_then(|v| v.as_table())
        .or_else(|| val.as_table());
    if let Some(tbl) = table {
        for (k, v) in tbl {
            if let Some(s) = v.as_str() {
                map.insert(k.clone(), s.to_string());
            }
        }
    }
    Ok(map)
}
