# FINAL PRODUCTION MODEL TRAINING & EVALUATION MODULE
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef

def build_and_save_final_model(X_train, X_test, y_train, y_test, optimal_indices):
    """
    Trains the final SVM strictly on the BPSO-selected feature dimensions,
    evaluates its empirical performance, and serializes the model to disk.
    """
    print("[*] Isolating the optimized feature space...")
    # Filter the matrices to ONLY include the swarm-selected features
    X_train_opt = X_train[:, optimal_indices]
    X_test_opt = X_test[:, optimal_indices]

    print(f"[*] Commencing full-scale training on {X_train_opt.shape[0]} atmospheric records...")
    print("    (This will take a few moments as the CPU processes the entire matrix)")

    # Instantiate the final max-margin engine
    final_svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42)
    final_svm.fit(X_train_opt, y_train)

    print("[*] Generating predictions against the unseen testing holdout set...")
    y_pred_final = final_svm.predict(X_test_opt)

    # Calculate empirical metrics for Chapter 4
    final_acc = accuracy_score(y_test, y_pred_final)
    final_mcc = matthews_corrcoef(y_test, y_pred_final)
    conf_matrix = confusion_matrix(y_test, y_pred_final)

    print("")
    print("             FINAL HYBRID MODEL PERFORMANCE            ")
    print("")
    print(f"[+] Final Accuracy : {final_acc * 100:.2f}%")
    print(f"[+] Final MCC      : {final_mcc:.4f}")
    print("\n[+] Confusion Matrix:")
    print(conf_matrix)
    print("\n[+] Detailed Classification Report:")
    print(classification_report(y_test, y_pred_final, target_names=["No Rain", "Rain"]))

    # Save the trained model parameters to disk for the Streamlit Dashboard
    joblib.dump(final_svm, 'bpso_svm_weather_model.pkl')
    joblib.dump(optimal_indices, 'optimal_feature_indices.pkl')
    print("\n[+] Model weights and feature indices successfully serialized to disk (.pkl).")
