from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, Binarizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from  pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from datetime import datetime as dt


spark = SparkSession.builder \
         .master("local[20]") \
         .appName("Lab 1 Exercise") \
         .config("spark.local.dir","/fastdata/acr21aa") \
         .getOrCreate()

 
sc = spark.sparkContext
sc.setLogLevel("WARN")

train = spark.read.load("../Data/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv", format = 'csv', inferSchema = True, header = False)

test = spark.read.load("../Data/XOR_Arbiter_PUFs/5xor_128bit/test_5xor_128dim.csv", format = 'csv', inferSchema = True, header = False)


train.printSchema()

#=================================Preprocessing=====================================
col_names = train.columns

# Chnaging the column datatypes for future use
for name in col_names:
	train = train.withColumn(name, col(name).cast("Double"))
	test = test.withColumn(name, col(name).cast("Double"))

train.printSchema()


# Creating smaller dataset for initial experiments
train_small = train.sample(0.01, seed = 6012).cache()
print("=========Number of samples in subset of train data==================")
print(train_small.count()) 

# Show count of missing or invalid values in all columns
print("Count of invalid or missing values for all the columns in train data")
print()
train_small.select([count(when(isnan(name) | col(name).isNull(), name)).alias(name) for name in col_names]).show()

# Combining all feature columns into a vector and mapping labels from [-1,1] to [0,1] 
vec_assembler = VectorAssembler(inputCols = col_names[:-1], outputCol = "feature_vector")
label_transformer = Binarizer(threshold = 0.0, inputCol = col_names[-1], outputCol = "labels")
 
evaluator = MulticlassClassificationEvaluator(labelCol = "labels", metricName = "accuracy")
evaluator2 = BinaryClassificationEvaluator(labelCol = "labels", metricName = "areaUnderROC")

'''
#==============================RANDOM FOREST=========================================
rf_classifier = RandomForestClassifier(featuresCol = "feature_vector", labelCol = "labels", maxDepth = 5, numTrees = 3, bootstrap = False, featureSubsetStrategy = 'all')

rf_stages = [vec_assembler, label_transformer, rf_classifier]
rf_pipeline	 = Pipeline(stages = rf_stages)

# random forest pipeline
rf_model = rf_pipeline.fit(train_small) 


# Cross-validator for random forest
paramGrid = ParamGridBuilder().addGrid(rf_classifier.maxDepth, [3, 5, 10]).addGrid(rf_classifier.numTrees, [3, 5, 10]).addGrid(rf_classifier.featureSubsetStrategy, ['all', 'sqrt', 'log2']).build()

cvRf = CrossValidator(estimator = rf_pipeline, estimatorParamMaps = paramGrid, evaluator = evaluator, numFolds = 5)

cvRf_model = cvRf.fit(train_small)
rf_best_params = {param[0].name: param[1] for param in cvRf_model.bestModel.stages[-1].extractParamMap().items()}


# ===================================LOGISTIC REGRESSION======================================
lr_classifier = LogisticRegression(featuresCol='feature_vector', labelCol='labels', maxIter=50, regParam=0.05, family="auto", elasticNetParam=0)

lr_stages = [vec_assembler, label_transformer, lr_classifier]
lr_pipeline = Pipeline(stages = lr_stages)

# logistic regression pipeline
lr_model = lr_pipeline.fit(train_small)


# Cross validator for logistic regression
paramGrid = ParamGridBuilder().addGrid(lr_classifier.maxIter, [30,50,100]).addGrid(lr_classifier.regParam, [0.01, 0.05, 0.1]).addGrid(lr_classifier.tol, [1e-3, 1e-4, 1e-5]).build()

cvLr = CrossValidator(estimator = lr_pipeline, estimatorParamMaps = paramGrid, evaluator = evaluator, numFolds = 5)
cvLr_model = cvLr.fit(train_small)

lr_best_params = {param[0].name: param[1] for param in cvLr_model.bestModel.stages[-1].extractParamMap().items()}
'''

# =============================NEURAL NETWORKS=================================================
layers = [len(col_names)-1, 40, 10, 2] 
nn_classifier = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="feature_vector", maxIter=100, layers=layers, seed=6012)

nn_stages = [vec_assembler, label_transformer, nn_classifier]
nn_pipeline = Pipeline(stages = nn_stages)


# neural network pipeline
nn_model = nn_pipeline.fit(train_small)


# Cross validator for neural network
paramGrid = ParamGridBuilder().addGrid(nn_classifier.maxIter, [30,50,100]).addGrid(nn_classifier.layers, [[len(col_names)-1, 20, 5, 2], [len(col_names)-1, 40, 10, 2], [len(col_names)-1, 50, 10, 2]]).addGrid(nn_classifier.stepSize, [0.03, 0.01, 0.06]).build()

cvNn = CrossValidator(estimator = nn_pipeline, estimatorParamMaps = paramGrid, evaluator = evaluator, numFolds = 5)
cvNn_model = cvNn.fit(train_small)


nn_best_params = {param[0].name: param[1] for param in cvNn_model.bestModel.stages[-1].extractParamMap().items()}


#========================PART-2============================================================
'''
# --------------RANDOM FOREST-------------------------------
rf_classifier = RandomForestClassifier(featuresCol = "feature_vector", labelCol = "labels", maxDepth = rf_best_params['maxDepth'], numTrees = rf_best_params['numTrees'], bootstrap = False, featureSubsetStrategy = rf_best_params['featureSubsetStrategy'])

rf_stages = [vec_assembler, label_transformer, rf_classifier]
rf_pipeline	 = Pipeline(stages = rf_stages)

# random forest pipeline
start = dt.now()
rf_model = rf_pipeline.fit(train)
rf_tR = (dt.now() - start).seconds  #training time
start = dt.now()
rf_predictions = rf_model.transform(test)
rf_ts = (dt.now() - start).seconds  # testing time
rf_accuracy = evaluator.evaluate(rf_predictions)
rf_auc = evaluator2.evaluate(rf_predictions)
print("====Random Forest Metrics=======")
print("Accuracy = %g " % rf_accuracy)
print("Area Under The Curve = %g " % rf_auc)
print("Training Time = %g " % rf_tR)
print("Testing Time = %g " % rf_ts)



#---------------LOGISTIC REGRESSION---------------------------
lr_classifier = LogisticRegression(featuresCol='feature_vector', labelCol='labels', maxIter=lr_best_params['maxIter'], regParam=lr_best_params['regParam'], family="auto", elasticNetParam=0, tol=lr_best_params['tol'])

lr_stages = [vec_assembler, label_transformer, lr_classifier]
lr_pipeline = Pipeline(stages = lr_stages)

# logistic regression pipeline
start = dt.now()
lr_model = lr_pipeline.fit(train)
lr_tR = (dt.now() - start).seconds
start = dt.now()
lr_predictions = lr_model.transform(test)
lr_ts = (dt.now() - start).seconds
lr_accuracy = evaluator.evaluate(lr_predictions)
lr_auc = evaluator2.evaluate(lr_predictions)


print("====Logistic regression Metrics=======")
print("Accuracy = %g " % lr_accuracy)
print("Area Under The Curve = %g " % lr_auc)
print("Training Time = %g " % lr_tR)
print("Testing Time = %g " % lr_ts)
'''

#-------------------------NEURAL NETWORK-----------------------
nn_classifier = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="feature_vector", maxIter=nn_best_params['maxIter'], layers=nn_best_params['layers'], seed=6012, stepSize=nn_best_params['stepSize'])

nn_stages = [vec_assembler, label_transformer, nn_classifier]
nn_pipeline = Pipeline(stages = nn_stages)


# neural network pipeline
start = dt.now()
nn_model = nn_pipeline.fit(train)
nn_tR = (dt.now() - start).seconds
start = dt.now()
nn_predictions = nn_model.transform(test)
nn_ts = (dt.now() - start).seconds
nn_accuracy = evaluator.evaluate(nn_predictions)
nn_auc = evaluator2.evaluate(nn_predictions)


print("====Neural Network Metrics=======")
print("Accuracy = %g " % nn_accuracy)
print("Area Under The Curve = %g " % nn_auc)
print("Training Time = %g " % nn_tR)
print("Testing Time = %g " % nn_ts)






