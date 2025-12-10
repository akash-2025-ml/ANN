from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd


model = tf.keras.models.load_model(
    r"C:\Users\INDIA\Desktop\open_cv\computer_Vision\ANN\ann_model_10-12-2025.h5"
)
# with open(
#     r"C:\Users\INDIA\Desktop\open_cv\computer_Vision\ANN\ANN_Model_9-12.pkl", "rb"
# ) as f:
#     model = pickle.load(f)
with open(
    r"C:\Users\INDIA\Desktop\open_cv\computer_Vision\ANN\label_encoder-output.pkl",
    "rb",
) as f:
    label_encoders_output = pickle.load(f)
########################################################################
categorical_columns = [
    "request_type",
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "tls_version",
    "ssl_validity_status",
    "unique_parent_process_names",
]

################################################################################
feature_names = [
    "sender_known_malicios",
    "sender_domain_reputation_score",
    "sender_spoof_detected",
    "sender_temp_email_likelihood",
    "dmarc_enforced",
    "packer_detected",
    "any_file_hash_malicious",
    "max_metadata_suspicious_score",
    "malicious_attachment_Count",
    "has_executable_attachment",
    "unscannable_attachment_present",
    "total_yara_match_count",
    "total_ioc_count",
    "max_behavioral_sandbox_score",
    "max_amsi_suspicion_score",
    "any_macro_enabled_document",
    "any_vbscript_javascript_detected",
    "any_active_x_objects_detected",
    "any_network_call_on_open",
    "max_exfiltration_behavior_score",
    "any_exploit_pattern_detected",
    "total_embedded_file_count",
    "max_suspicious_string_entropy_score",
    "max_sandbox_execution_time",
    "unique_parent_process_names",
    "return_path_mismatch_with_from",
    "return_path_known_malicious",
    "return_path_reputation_score",
    "reply_path_known_malicious",
    "reply_path_diff_from_sender",
    "reply_path_reputation_Score",
    "smtp_ip_known_malicious",
    "smtp_ip_geo",
    "smtp_ip_asn",
    "smtp_ip_reputation_score",
    "domain_known_malicious",
    "url_Count",
    "dns_morphing_detected",
    "domain_tech_stack_match_score",
    "is_high_risk_role_targeted",
    "sender_name_similarity_to_vip",
    "urgency_keywords_present",
    "request_type",
    "content_spam_score",
    "user_marked_as_spam_before",
    "bulk_message_indicator",
    "unsubscribe_link_present",
    "marketing-keywords_detected",
    "html_text_ratio",
    "image_only_email",
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "reverse_dns_valid",
    "tls_version",
    "total_links_detected",
    "url_shortener_detected",
    "url_redirect_chain_length",
    "final_url_known_malicious",
    "url_decoded_spoof_detected",
    "url_reputation_score",
    "ssl_validity_status",
    "site_visual_similarity_to_known_brand",
    "url_rendering_behavior_score",
    "link_rewritten_through_redirector",
    "token_validation_success",
    "total_components_detected_malicious",
    "Analysis_of_the_qrcode_if_present",
]


##############################################################################################
default_values = {col: 0 for col in feature_names}  # default numeric value

default_values.update(
    {
        "sender_known_malicios": 0,
        "sender_domain_reputation_score": 0.95,
        "sender_spoof_detected": 0,
        "sender_temp_email_likelihood": 0.0,
        "dmarc_enforced": 1,
        "packer_detected": 0,
        "any_file_hash_malicious": 0,
        "max_metadata_suspicious_score": 0.0,
        "malicious_attachment_Count": 0,
        "has_executable_attachment": 0,
        "unscannable_attachment_present": 0,
        "total_yara_match_count": 0,
        "total_ioc_count": 0,
        "max_behavioral_sandbox_score": 0.0,
        "max_amsi_suspicion_score": 0.0,
        "any_macro_enabled_document": 0,
        "any_vbscript_javascript_detected": 0,
        "any_active_x_objects_detected": 0,
        "any_network_call_on_open": 0,
        "max_exfiltration_behavior_score": 0.0,
        "any_exploit_pattern_detected": 0,
        "total_embedded_file_count": 0,
        "max_suspicious_string_entropy_score": 0.075607,
        "max_sandbox_execution_time": 4.377776483e-107,
        "unique_parent_process_names": '[""]',
        "return_path_mismatch_with_from": 0,
        "return_path_known_malicious": 0,
        "return_path_reputation_score": 0.95,
        "reply_path_known_malicious": 0,
        "reply_path_diff_from_sender": 0,
        "reply_path_reputation_Score": 0.95,
        "smtp_ip_known_malicious": 0,
        "smtp_ip_geo": 0.001,
        "smtp_ip_asn": 0.05,
        "smtp_ip_reputation_score": 0.95,
        "domain_known_malicious": 0,
        "url_Count": 0,
        "dns_morphing_detected": 0,
        "domain_tech_stack_match_score": 1.0,
        "is_high_risk_role_targeted": 0,
        "sender_name_similarity_to_vip": 0.0000000,
        "urgency_keywords_present": 0,
        "request_type": "none",
        "content_spam_score": 0.0,
        "user_marked_as_spam_before": 0,
        "bulk_message_indicator": 0,
        "unsubscribe_link_present": 0,
        "marketing-keywords_detected": 0,
        "html_text_ratio": 0.0,
        "image_only_email": 0,
        "spf_result": "pass",
        "dkim_result": "pass",
        "dmarc_result": "pass",
        "reverse_dns_valid": 1,
        "tls_version": "TLS 1.0",
        "total_links_detected": 0,
        "url_shortener_detected": 0,
        "url_redirect_chain_length": 0,
        "final_url_known_malicious": 0,
        "url_decoded_spoof_detected": 0,
        "url_reputation_score": 0.0,
        "ssl_validity_status": "valid",
        "site_visual_similarity_to_known_brand": 0.0,
        "url_rendering_behavior_score": 0.0000430629,
        "link_rewritten_through_redirector": 0,
        "token_validation_success": 1,
        "total_components_detected_malicious": 0,
        "Analysis_of_the_qrcode_if_present": 2,
    }
)


with open("le_dkim_result.pkl", "rb") as f:
    le_dkim_result = pickle.load(f)

with open("le_dmarc_result.pkl", "rb") as f1:
    le_dmarc_result = pickle.load(f1)

with open("le_request_type.pkl", "rb") as f2:
    le_request_type = pickle.load(f2)

with open("le_spf_result.pkl", "rb") as f3:
    le_spf_result = pickle.load(f3)

with open("le_ssl_validity_status.pkl", "rb") as f4:
    le_ssl_validity_status = pickle.load(f4)

with open("le_tls_version.pkl", "rb") as f5:
    le_tls_version = pickle.load(f5)

with open("le_unique_parent_process_names.pkl", "rb") as f6:
    le_unique_parent_process_names = pickle.load(f6)
svd = joblib.load(r"C:\Users\INDIA\Desktop\open_cv\computer_Vision\ANN\svd.pkl")


def predict_1(input_data):
    """
    Improved predict function using encode_dataframe
    """
    # data_dict = input_data["data"]
    data_dict = input_data.data

    # Create a single-row DataFrame from input data
    df = pd.DataFrame([data_dict])
    # print(df.head())

    # Fill missing values with defaults
    for col in feature_names:
        if col not in df.columns:
            df[col] = default_values.get(col, 0)

    try:
        df["request_type"] = le_request_type.transform(df["request_type"])
    except:

        df["request_type"] = 10  # none

    try:
        df["spf_result"] = le_spf_result.transform(df["spf_result"])
    except:

        df["spf_result"] = 2  # none

    try:
        df["dkim_result"] = le_dkim_result.transform(df["dkim_result"])

    except:

        df["dkim_result"] = 2  # none

    try:
        df["dmarc_result"] = le_dmarc_result.transform(df["dmarc_result"])
    except:
        df["dmarc_result"] = 2  # none

    try:
        df["tls_version"] = le_tls_version.transform(df["tls_version"])
    except:
        df["tls_version"] = 4  # TLS 1.2

    try:
        df["ssl_validity_status"] = le_ssl_validity_status.transform(
            df["ssl_validity_status"]
        )
    except:
        df["ssl_validity_status"] = 8  # valid

    try:
        df["unique_parent_process_names"] = le_unique_parent_process_names.transform(
            df["unique_parent_process_names"]
        )
    except:
        df["unique_parent_process_names"] = 0  # "[""]"

    # print("new === ",df.head())

    # Convert "true"/"false" strings to 1/0
    for col in df.columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: (
                    1
                    if isinstance(x, str) and x.lower() == "true"
                    else 0 if isinstance(x, str) and x.lower() == "false" else x
                )
            )

    row = df[feature_names].iloc[0].tolist()

    # return row

    #   # Make 2D array
    final_input = [row]
    # print("final_inpu = ",final_input)
    # Apply SVD
    X_svd = svd.transform(final_input)

    # Predict
    # prediction = model.predict(X_svd)
    probs = model.predict(X_svd)
    probs = probs.tolist()
    pred_class = np.argmax(probs, axis=1)
    # print("pred_class === ",pred_class)
    # print("Class === ",label_encoders_output.inverse_transform(pred_class)[0])
    # print("Encoder output keys:", label_encoders_output.keys())
    # return label_encoders_output["target"].inverse_transform(pred_class)[0]
    print(str(pred_class))
    decoded_label = label_encoders_output.inverse_transform(pred_class)[0]

    return decoded_label, probs


app = FastAPI()
# l = []


# Input schema
class InputData(BaseModel):
    data: dict  # {"feature_name": value, ...}


@app.get("/")
def home():
    return {"message": "Welcome to the ML API! Send POST to /predict with JSON data."}


@app.post("/predict")
def predict(input_data: InputData):
    output, a = predict_1(input_data)

    return {
        "predicted_class": output,
        "Malicious": a[0][0],
        "Warning": a[0][3],
        "No Action": a[0][1],
        "spam": a[0][2],
    }


# input fastapi
# uvicorn main:app --reload
# {
#   "data": {"spf_result": "pass",
#   "dkim_result": "pass",
#   "tls_version": "not available",
#   "dmarc_result": "pass",
#   "reverse_dns_valid": 0,
#   "sender_temporary_email_likelihood": 1.0,
#   "urgency_keywords_present": 0.0,
#   "sender_known_malicious": 0,
#   "dmarc_enforced": 0,
#   "smtp_ip_known_malicious": 0,
#   "is_high_risk_role_targeted": 0,
#   "sender_name_similarity_to_vip": 0,
#   "return_path_known_malicious": 0,
#   "ssl_validity_status": "valid",
#   "total_links_detected": 2,
#   "token_validation_success": 0,
#   "final_url_known_malicious": 0,
#   "link_rewritten_through_redirector": 0,
#   "site_visual_similarity_to_known_brand": 0.0,
#   "url_count": 0,
#   "url_reputation_score": 1.0,
#   "dns_morphing_detected": false,
#   "url_shortener_detected": false,
#   "url_redirect_chain_length": 0,
#   "url_decoded_spoof_detected": false,
#   "domain_tech_stack_match_score": 0.25,
#   "html_text_ratio": 0.10677906568317529,
#   "image_only_email": false,
#   "bulk_message_indicator": false,
#   "unsubscribe_link_present": false,
#   "marketing_keywords_detected": 0.04081632653061225,
#   "domain_known_malicious": 0,
#   "return_path_mismatch_with_from": 0,
#   "sender_spoof_detected": 0,
#   "url_rendering_behavior_score": 0.0,
#   "user_marked_as_spam_before": 0,
#   "analysis_of_qr_if_present": 2,
#   "reply_path_reputation_score": 1,
#   "reply_path_known_malicious": 0,
#   "reply_path_diff_from_sender": 0,
#   "smtp_ip_geo": 0.25,
#   "smtp_ip_asn": 0.5,
#   "return_path_reputation_score": 1.0}
# }
