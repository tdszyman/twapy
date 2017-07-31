from twapy import Evaluation, score_evaluation_file

evals_fn = "groundtruth.csv"
models_dir = "models"

# This will take a sample of 10 model pairs and predict the temporal word analogies for all the
# labeled examples in those pairs of years, saving the prediction output to a file. Then,
# that file is loaded and scored to give the accuracy results.
e = Evaluation(evals_fn, models_dir, samplesize=0.5, output_fn="evaluation_output.tsv")
#e.evaluate_sample(10)
e.evaluate_all()
score_evaluation_file("evaluation_output.tsv")
