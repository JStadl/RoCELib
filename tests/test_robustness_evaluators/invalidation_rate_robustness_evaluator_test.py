from rocelib.evaluations.robustness_evaluations.NE_Robustness_Implementations.InvalidationRateRobustnessEvaluator import \
    InvalidationRateRobustnessEvaluator


def test_ionosphere_kdtree_robustness(testing_models):
    method = "KDTreeNNCE"
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    ct.generate([method])
    opt = InvalidationRateRobustnessEvaluator(ct)

    robusts = opt.evaluate(method)
    print(f"Percentage of Robust CEs: {sum(robusts)/len(robusts) * 100:.2f}%")
