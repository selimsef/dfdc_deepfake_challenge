TEST_DIR=$1
CSV=$2

python predict_folder.py \
 --test-dir $TEST_DIR \
 --output $CSV \
 --models final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36 \
  final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19 \
  final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29 \
  final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31 \
  final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37 \
  final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40 \
  final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23