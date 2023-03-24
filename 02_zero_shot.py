# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC Brainstorming on use cases
# MAGIC Objectives
# MAGIC 1. Shorten calls
# MAGIC 2. Increase customer satisfication
# MAGIC 
# MAGIC * What does a call center want to do? Bryan and Avi to chat with PetSmart and explore requirements

# COMMAND ----------

# MAGIC %pip install --upgrade transformers sacremoses

# COMMAND ----------

# MAGIC %run ./util/notebook-config

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Offline analysis
# MAGIC * staffing decision support: topic, duration (non-interactive, after-call analysis) 
# MAGIC * end of shift summary to evaluate performance (how happy are customers? breakdown of performance on topics, post-mortem)

# COMMAND ----------

df = spark.read.table("bergmann_transcriptions")
pandas_df = df.toPandas()
pandas_df

# COMMAND ----------

# DBTITLE 1,Example of zero-shot classification for topic modeling
import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(model="facebook/bart-large-mnli",
  device=device, 
  chunk_length_s=30,  
)

# Define the topics we are interested in identifying
### fire, crime, animal, other -- Nicoles' German is at college 201 level so definitely needs a native's pair of eyes
classes = ['Feuer', 'Verbrechen', 'Tier', 'Andere'] 

# COMMAND ----------

# DBTITLE 1,Take one of the transcribed example and run it through the zero-shot classification pipeline
pipe("""
Die Feuerwehr! - Feuerwehr! - Ja, fahrt sofort Dingelsdorf, Ortsausgang Richtung Litzelstetten, Ortsausgang Richtung Dingelsdorf, schwerer VU, personisch eingeklemmt. - Jawohl, gut, merci, Herr Turm, bis zum Ort! - Ade!""",
    candidate_labels=classes, 
)

# COMMAND ----------

# MAGIC %md ## Scaling up zero-shot classification
# MAGIC 
# MAGIC Similar to how we scaled up transcription in notebook 01, we use a pandas UDF. Because of the pipeline's size, we need to broadcast the pipeline to each worker and retrieve the broadcasted pipeline within the UDF.

# COMMAND ----------

broadcast_pipeline = spark.sparkContext.broadcast(pipe)

# COMMAND ----------

# DBTITLE 1,Define UDF for topic modeling
import pandas as pd
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StructType, StructField, ArrayType, StringType, FloatType

schema = StructType([	
  StructField("labels",ArrayType(StringType(), True),True), 	
  StructField("scores", ArrayType(FloatType(), True), True)])

@pandas_udf(schema)
def topic_modeling_udf(sequences: pd.Series) -> pd.DataFrame:
  pipe = broadcast_pipeline.value
  outputs = pipe(sequences.to_list()
      , candidate_labels=['Feuer', 'Verbrechen', 'Tier', 'Andere']  # fire, crime, animal, medical emergency, other -- Nicoles' German is at college 201 level so definitely needs a native's pair of eyes
      , batch_size=1
  ) 

  class_scores = [{"labels": o["labels"], "scores": o["scores"]} for o in outputs]
  return pd.DataFrame(class_scores)

# COMMAND ----------

# DBTITLE 1,To make the analysis easier to evaluate for the English audience, we also translate the script into English
from transformers import pipeline
ge_en_translator = pipeline("translation_de_to_en", "facebook/wmt19-de-en")
broadcast_ge_en = spark.sparkContext.broadcast(ge_en_translator)

@pandas_udf("string")
def translate_udf(sequences: pd.Series) -> pd.Series:
  pipe = broadcast_ge_en.value
  translation_text = [o["translation_text"] for o in pipe(sequences.to_list(), max_length=400) ]
  return pd.Series(translation_text)

# COMMAND ----------

# DBTITLE 1,Apply the UDFs on our source dataset
topic_modelled = (df
  .withColumn("topic", topic_modeling_udf(col("transcription")))
  .withColumn("top_class", col("topic.labels")[0])
  .withColumn("translate_EN", translate_udf(col("transcription")))
)

topic_modelled.cache()
display(topic_modelled)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Any other use cases? 

# COMMAND ----------


