"""
Test script for F1 score metric calculation
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.evaluator import calculate_metrics


def create_result(overall_accuracy, p_value_within_tolerance, test_method_correct,
                   gt_decision, llm_decision):
    """Helper to create a result in the actual format used by ht.py"""
    return {
        "evaluation": {
            "overall_accuracy": overall_accuracy,
            "p_value": {"within_tolerance": p_value_within_tolerance},
            "test_method": 1.0 if test_method_correct else 0.0,
            "decision": {"predicted": llm_decision}
        },
        "ground_truth": {
            "decision": gt_decision
        }
    }


def test_f1_perfect_score():
    """Test F1 score with perfect predictions"""
    results = [
        create_result(1.0, True, True, "reject_H0", "reject_H0"),
        create_result(1.0, True, True, "fail_to_reject_H0", "fail_to_reject_H0"),
        create_result(1.0, True, True, "reject_H0", "reject_H0"),
    ]
    
    metrics = calculate_metrics(results)
    
    print("✅ Perfect Score Test:")
    print(f"   Accuracy: {metrics['accuracy']:.2%}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1 Score: {metrics['f1_score']:.3f}")
    print(f"   TP={metrics['confusion_matrix']['true_positives']}, "
          f"FP={metrics['confusion_matrix']['false_positives']}, "
          f"FN={metrics['confusion_matrix']['false_negatives']}, "
          f"TN={metrics['confusion_matrix']['true_negatives']}")
    
    assert metrics['f1_score'] == 1.0, f"Expected perfect F1 score, got {metrics['f1_score']}"
    assert metrics['precision'] == 1.0, f"Expected perfect precision, got {metrics['precision']}"
    assert metrics['recall'] == 1.0, f"Expected perfect recall, got {metrics['recall']}"
    print("   ✓ All assertions passed\n")


def test_f1_with_false_positives():
    """Test F1 score with false positives (model over-rejects)"""
    results = [
        # TP: correctly reject
        create_result(1.0, True, True, "reject_H0", "reject_H0"),
        # FP: incorrectly reject
        create_result(0.5, False, True, "fail_to_reject_H0", "reject_H0"),
        # TN: correctly not reject
        create_result(1.0, True, True, "fail_to_reject_H0", "fail_to_reject_H0"),
    ]
    
    metrics = calculate_metrics(results)
    
    print("⚠️  False Positive Test (Over-rejection):")
    print(f"   Accuracy: {metrics['accuracy']:.2%}")
    print(f"   Precision: {metrics['precision']:.3f} (TP=1, FP=1)")
    print(f"   Recall: {metrics['recall']:.3f} (TP=1, FN=0)")
    print(f"   F1 Score: {metrics['f1_score']:.3f}")
    
    assert metrics['precision'] == 0.5, f"Expected precision 0.5, got {metrics['precision']}"
    assert metrics['recall'] == 1.0, f"Expected recall 1.0, got {metrics['recall']}"
    assert abs(metrics['f1_score'] - 0.667) < 0.01, f"Expected F1 ~0.667, got {metrics['f1_score']}"
    print("   ✓ All assertions passed\n")


def test_f1_with_false_negatives():
    """Test F1 score with false negatives (model under-rejects)"""
    results = [
        # TP: correctly reject
        create_result(1.0, True, True, "reject_H0", "reject_H0"),
        # FN: should reject but didn't
        create_result(0.5, False, True, "reject_H0", "fail_to_reject_H0"),
        # TN: correctly not reject
        create_result(1.0, True, True, "fail_to_reject_H0", "fail_to_reject_H0"),
    ]
    
    metrics = calculate_metrics(results)
    
    print("⚠️  False Negative Test (Under-rejection):")
    print(f"   Accuracy: {metrics['accuracy']:.2%}")
    print(f"   Precision: {metrics['precision']:.3f} (TP=1, FP=0)")
    print(f"   Recall: {metrics['recall']:.3f} (TP=1, FN=1)")
    print(f"   F1 Score: {metrics['f1_score']:.3f}")
    
    assert metrics['precision'] == 1.0, f"Expected precision 1.0, got {metrics['precision']}"
    assert metrics['recall'] == 0.5, f"Expected recall 0.5, got {metrics['recall']}"
    assert abs(metrics['f1_score'] - 0.667) < 0.01, f"Expected F1 ~0.667, got {metrics['f1_score']}"
    print("   ✓ All assertions passed\n")


def test_f1_balanced_errors():
    """Test F1 score with both FP and FN"""
    results = [
        # TP
        create_result(1.0, True, True, "reject_H0", "reject_H0"),
        create_result(1.0, True, True, "reject_H0", "reject_H0"),
        # FP
        create_result(0.5, False, True, "fail_to_reject_H0", "reject_H0"),
        # FN
        create_result(0.5, False, True, "reject_H0", "fail_to_reject_H0"),
        # TN
        create_result(1.0, True, True, "fail_to_reject_H0", "fail_to_reject_H0"),
    ]
    
    metrics = calculate_metrics(results)
    
    print("⚖️  Balanced Errors Test:")
    print(f"   Accuracy: {metrics['accuracy']:.2%}")
    print(f"   Precision: {metrics['precision']:.3f} (TP=2, FP=1)")
    print(f"   Recall: {metrics['recall']:.3f} (TP=2, FN=1)")
    print(f"   F1 Score: {metrics['f1_score']:.3f}")
    
    assert abs(metrics['precision'] - 0.667) < 0.01, f"Expected precision ~0.667, got {metrics['precision']}"
    assert abs(metrics['recall'] - 0.667) < 0.01, f"Expected recall ~0.667, got {metrics['recall']}"
    assert abs(metrics['f1_score'] - 0.667) < 0.01, f"Expected F1 ~0.667, got {metrics['f1_score']}"
    print("   ✓ All assertions passed\n")


def test_f1_edge_case_no_rejections():
    """Test F1 when model never rejects (edge case)"""
    results = [
        create_result(0.5, False, True, "reject_H0", "fail_to_reject_H0"),
        create_result(1.0, True, True, "fail_to_reject_H0", "fail_to_reject_H0"),
    ]
    
    metrics = calculate_metrics(results)
    
    print("🚫 Edge Case - No Rejections:")
    print(f"   Accuracy: {metrics['accuracy']:.2%}")
    print(f"   Precision: {metrics['precision']:.3f} (TP=0, FP=0)")
    print(f"   Recall: {metrics['recall']:.3f} (TP=0, FN=1)")
    print(f"   F1 Score: {metrics['f1_score']:.3f}")
    
    assert metrics['precision'] == 0.0, "Expected precision 0 when no predictions"
    assert metrics['recall'] == 0.0, "Expected recall 0 when missing all positives"
    assert metrics['f1_score'] == 0.0, "Expected F1 score 0"
    print("   ✓ All assertions passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing F1 Score Metric Implementation")
    print("=" * 60 + "\n")
    
    try:
        test_f1_perfect_score()
        test_f1_with_false_positives()
        test_f1_with_false_negatives()
        test_f1_balanced_errors()
        test_f1_edge_case_no_rejections()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nF1 score metric is working correctly.")
        print("You can now run a real benchmark to see it in action:")
        print("  python ht.py --mode quick")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
