import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import Any, Dict, List

# Set page title
st.title("Evaluador de Predicciones de Atributos de SKU")

# Helper functions from the original code
def normalize_value(value):
    """Normalize values for consistent comparison."""
    if value is None:
        return ""
    
    # Convert to string
    value_str = str(value).strip().lower()
        
    return value_str

def is_match(pred_value: Any, true_value: Any) -> bool:
    """
    Determine if a predicted value matches a ground truth value.
    Match occurs if the ground truth value is contained within the predicted value.
    """
    pred_norm = normalize_value(pred_value)
    true_norm = normalize_value(true_value)
    
    # Exact match check
    if pred_norm == true_norm:
        return True
        
    # Check if ground truth is contained in prediction
    if true_norm in pred_norm:
        return True
        
    return False

def evaluate_sku_with_placemarker(predicted, ground_truth):
    """
    Evaluate a single SKU's attribute predictions against ground truth.
    
    Args:
        predicted: Dictionary of attribute name -> predicted value
        ground_truth: Dictionary of attribute name -> true value
        
    Returns:
        Dictionary with evaluation metrics for this SKU
    """
    all_keys = set(ground_truth.keys()).union(set(predicted.keys()))
    
    # Initialize counters for found vs correct confusion matrix
    found_correct = 0      # Attribute found and value is correct
    found_incorrect = 0    # Attribute found but value is incorrect
    not_found = 0          # Attribute not found at all
    extra_attributes = 0   # Attributes found that don't exist in ground truth
    
    # Check each attribute
    per_attribute_results = {}
    for key in all_keys:
        pred_value = predicted.get(key)
        true_value = ground_truth.get(key)
        
        # Skip if ground truth doesn't have this attribute
        if true_value is None:
            if pred_value is not None and pred_value != "xxx":
                extra_attributes += 1  # Extra attribute found
            continue
            
        # Check if prediction has this attribute
        if pred_value is None or pred_value == "xxx":
            not_found += 1  # Attribute not found
            per_attribute_results[key] = {"found": False, "correct": False}
            continue
            
        # Both have the attribute, check if values match
        is_correct = is_match(pred_value, true_value)
        
        if is_correct:
            found_correct += 1  # Found and correct
            per_attribute_results[key] = {"found": True, "correct": True, "pred": pred_value, "true": true_value}
        else:
            found_incorrect += 1  # Found but incorrect
            per_attribute_results[key] = {"found": True, "correct": False, "pred": pred_value, "true": true_value}
    
    # Calculate traditional metrics using new counters
    total_ground_truth_attrs = len(ground_truth)
    total_found = found_correct + found_incorrect
    
    # Calculate metrics
    precision = found_correct / (total_found + extra_attributes) if (total_found + extra_attributes) > 0 else 0
    recall = found_correct / total_ground_truth_attrs if total_ground_truth_attrs > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate new metrics
    attribute_coverage = total_found / total_ground_truth_attrs if total_ground_truth_attrs > 0 else 0
    attribute_correctness = found_correct / total_ground_truth_attrs if total_ground_truth_attrs > 0 else 0
    attribute_accuracy = found_correct / total_found if total_found > 0 else 0
    f2 = 2 * attribute_accuracy * attribute_correctness / (attribute_accuracy + attribute_correctness) if (attribute_accuracy + attribute_correctness) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "attr_coverage": attribute_coverage,
        "attr_correctness": attribute_correctness,
        "attr_accuracy": attribute_accuracy,
        "f2": f2,
        "attr_expected": total_ground_truth_attrs,
        "attr_found": total_found,
        "not_found": not_found,
        "found_correct": found_correct,
        "found_incorrect": found_incorrect,
        "extra_attributes": extra_attributes,
        "per_attribute": per_attribute_results
    }

def evaluate_dataset(predictions, ground_truth):
    """
    Evaluate attribute predictions for multiple SKUs.
    
    Args:
        predictions: Dictionary of SKU -> attribute dictionary
        ground_truth: Dictionary of SKU -> attribute dictionary
        
    Returns:
        Dictionary with aggregate evaluation metrics
    """
    all_skus = set(ground_truth.keys())
    
    # Track metrics for all SKUs
    sku_metrics = {}
    attribute_level_results = []
    
    # Calculate metrics for each SKU
    for sku in all_skus:
        if sku not in predictions:
            sku_metrics[sku] = {
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "attr_coverage": 0,
                "attr_correctness": 0,
                "f2": 0,
                "attr_accuracy": 0,
                "attr_found": 0,
                "attr_expected": len(ground_truth[sku]),
                "not_found": len(ground_truth[sku]),
                "found_correct": 0,
                "found_incorrect": 0,
                "extra_attributes": 0,
                "sku_found": False
            }
            continue
            
        # Evaluate this SKU
        sku_result = evaluate_sku_with_placemarker(predictions[sku], ground_truth[sku])
        sku_result["sku_found"] = True
        sku_metrics[sku] = sku_result
        
        # Collect attribute-level data for more detailed analysis
        for attr, result in sku_result["per_attribute"].items():
            attribute_level_results.append({
                "sku": sku,
                "attribute": attr,
                "found": result["found"],
                "correct": result["correct"],
                "predicted": result.get("pred"),
                "ground_truth": result.get("true")
            })
    
    # Calculate aggregate metrics
    precision_values = [m["precision"] for m in sku_metrics.values()]
    recall_values = [m["recall"] for m in sku_metrics.values()]
    f1_values = [m["f1"] for m in sku_metrics.values()]
    attribute_coverage_values = [m["attr_coverage"] for m in sku_metrics.values()]
    attribute_correctness_values = [m["attr_correctness"] for m in sku_metrics.values()]
    attribute_accuracy_values = [m.get("attr_accuracy", 0) for m in sku_metrics.values() if m.get("attr_found", 0) > 0]
    f2_values = [m["f2"] for m in sku_metrics.values()]
    
    # Create metrics dataframe for easier analysis
    metrics_df = pd.DataFrame(sku_metrics).T
    attr_results_df = pd.DataFrame(attribute_level_results)
    
    # Calculate metrics by attribute type
    attribute_type_metrics = {}
    if not attr_results_df.empty:
        for attr, group in attr_results_df.groupby("attribute"):
            attribute_type_metrics[attr] = {
                "count": len(group),
                "found_rate": group["found"].mean(),
                "accuracy": group["correct"].mean() if group["found"].sum() > 0 else 0
            }
    
    # Aggregate results
    return {
        "overall": {
            "mean_precision": np.mean(precision_values),
            "mean_recall": np.mean(recall_values),
            "mean_f1": np.mean(f1_values),
            "mean_attr_coverage": np.mean(attribute_coverage_values),
            "mean_attr_correctness": np.mean(attribute_correctness_values),
            "mean_attr_accuracy": np.mean(attribute_accuracy_values) if attribute_accuracy_values else 0,
            "mean_f2": np.mean(f2_values),
            "sku_coverage": sum(1 for m in sku_metrics.values() if m.get("sku_found", False)) / len(all_skus)
        },
        "per_sku": sku_metrics,
        "per_attribute_type": attribute_type_metrics,
        "sku_metrics_df": metrics_df,
        "attribute_results_df": attr_results_df
    }

# Modified function to load from text files containing JSON
def load_ground_truth_and_predictions(gt_file, pred_file):
    """
    Load ground truth and predictions from uploaded .txt files containing JSON
    
    Args:
        gt_file: Ground truth JSON text file
        pred_file: Predictions JSON text file
        
    Returns:
        Tuple of (ground_truth, predictions) dictionaries
    """
    try:
        # Read the content of the uploaded text files
        gt_content = gt_file.read().decode('utf-8')
        pred_content = pred_file.read().decode('utf-8')
        
        # Parse JSON from the content
        ground_truth = json.loads(gt_content)
        predictions = json.loads(pred_content)
        
        return ground_truth, predictions
    
    except json.JSONDecodeError as e:
        st.error(f"Error al analizar JSON: {str(e)}")
        return None, None
    except Exception as e:
        st.error(f"Error al cargar archivos: {str(e)}")
        return None, None

# Streamlit app layout
st.markdown("""
Esta aplicación evalúa el rendimiento (accuracy) de las predicciones (extracciones) de atributos contra los datos de la Fuente de la Verdad de cada SKU.
Subir 2 archivos (texto en formato json): verdad y predicciones para ver las métricas de evaluación.
""")

# File uploaders
st.header("Subir Archivos de Datos")
gt_file = st.file_uploader("Subir Archivo de Texto de la Verdad", type="txt")
pred_file = st.file_uploader("Subir Archivo de Texto de Predicciones", type="txt")

# Information about the expected file format
with st.expander("Información del Formato de Archivo de Texto"):
    st.markdown("""
    Tus archivos de texto deben contener datos JSON en el siguiente formato:
    
    ```json
    {
        "SKU1": {
            "attribute1": "value1",
            "attribute2": "value2"
        },
        "SKU2": {
            "attribute1": "value3",
            "attribute3": "value4"
        }
    }
    ```
    
    Cada archivo debe tener los identificadores SKU como claves de nivel superior, con un diccionario anidado
    de pares atributo-valor para cada SKU.
    """)

# Process uploaded files
if gt_file is not None and pred_file is not None:
    with st.spinner("Procesando archivos y evaluando predicciones..."):
        # Load data from text files (JSON)
        ground_truth, predictions = load_ground_truth_and_predictions(gt_file, pred_file)
        
        if ground_truth and predictions:
            # Run evaluation
            results = evaluate_dataset(predictions=predictions, ground_truth=ground_truth)
            
            # Display results
            st.header("Resultados de Evaluación")
            
            # 1. Global Results
            st.subheader("Métricas Globales")
            results_global = pd.DataFrame(list(results['overall'].items()), columns=['KPI', 'Valor'])
            st.dataframe(results_global)
            
            # 2. SKU Results
            st.subheader("Resultados por SKU")
            keep_cols = ['attr_coverage', 'attr_correctness', 'attr_accuracy', 'f2',
                         'attr_expected', 'attr_found', 'not_found', 'found_correct',
                         'found_incorrect', 'extra_attributes']
            results_sku = results['sku_metrics_df'][keep_cols]
            st.dataframe(results_sku)
            
            # 3. SKU Attribute Results
            st.subheader("Resultados por Atributo de Atributos por SKU")
            results_sku_attr = results['attribute_results_df']
            st.dataframe(results_sku_attr)
            
            # 4. Attribute Group Results
            st.subheader("Resultados Agrupados por Atributo")
            grouped = results_sku_attr.groupby('attribute')
            result_attr = grouped.apply(
                lambda group: pd.Series({
                    '%_encontrado_correcto_TT': ((group['found'] == True) & (group['correct'] == True)).mean().round(2),
                    '%_encontrado_correcto_TF': ((group['found'] == True) & (group['correct'] == False)).mean().round(2),
                    '%_encontrado_F': (group['found'] == False).mean().round(2)
                })
            ).reset_index()
            st.dataframe(result_attr)
            
            # Download buttons
            st.header("Descargar Resultados")
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            
            # Helper function for CSV conversion
            def convert_df_to_csv(df):
                return df.to_csv().encode('utf-8')
            
            with col1:
                st.download_button(
                    label="Descargar Resultados Globales",
                    data=convert_df_to_csv(results_global),
                    file_name='resultados_globales.csv',
                    mime='text/csv',
                )
            
            with col2:
                st.download_button(
                    label="Descargar Resultados de SKU",
                    data=convert_df_to_csv(results_sku),
                    file_name='resultados_sku.csv',
                    mime='text/csv',
                )
            
            with col3:
                st.download_button(
                    label="Descargar Resultados de Atributos por SKU",
                    data=convert_df_to_csv(results_sku_attr),
                    file_name='resultados_atributos.csv',
                    mime='text/csv',
                )
                
            with col4:
                st.download_button(
                    label="Descargar Resultados Agrupados por Atributo",
                    data=convert_df_to_csv(result_attr),
                    file_name='resultados_atributos_agrupados.csv',
                    mime='text/csv',
                )
else:
    st.info("Por favor, sube los archivos de texto de verdad fundamental y predicciones para comenzar la evaluación.")