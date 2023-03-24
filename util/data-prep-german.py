# Databricks notebook source
# MAGIC %run ./notebook-config

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We download the datasets by Bergmann, J. (1993). Alarmiertes verstehen: Kommunikation in Feuerwehrnotrufen. Wirklichkeit im Deutungsprozeß. In T. Jung & S. Müller-Doohm (Eds.), Verstehen und Methoden in den Kultur-und Sozialwissenschaften (pp. 283-328). Frankfurt aM: Suhrkamp.

# COMMAND ----------

# DBTITLE 1,Download source data
# MAGIC %sh -e
# MAGIC cd /databricks/driver
# MAGIC wget -e robots=off -R "index.html*" -N -nH -l inf -r --no-parent https://media.talkbank.org/ca/Bergmann/
# MAGIC cp -r ca/Bergmann/ /dbfs/tutorials

# COMMAND ----------

dbutils.fs.ls("/tutorials/Bergmann/")

# COMMAND ----------

# DBTITLE 1,Take a look at the path structure of our source data files
import os
from glob import glob
audio_files = [y for x in os.walk("/dbfs/tutorials/Bergmann/") for y in glob(os.path.join(x[0], '*.mp3'))]
print(audio_files[:10])

# COMMAND ----------

# DBTITLE 1,Create path dataframe
import pandas as pd
pandas_df = pd.DataFrame(pd.Series(audio_files),columns=["path"])
df = spark.createDataFrame(pandas_df)
display(df.limit(10))

# COMMAND ----------

# DBTITLE 1,Add unique id and write to path table
df_with_ids = df.selectExpr("path", "uuid() as id")
df_with_ids.write.mode("overwrite").saveAsTable("bergmann_paths_with_ids")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create binary table

# COMMAND ----------

binary_df = spark.readStream.format("cloudFiles") \
  .option("cloudFiles.format", "binaryFile") \
  .option("recursiveFileLookup", "true") \
  .option("pathGlobFilter", "*.mp3") \
  .load("/tutorials/Bergmann") \
  .repartition(32)

# COMMAND ----------

binary_df = binary_df.selectExpr("*", "uuid() as id")

# COMMAND ----------

binary_df.writeStream.format("delta")\
  .option("checkpointLocation", config["checkpoint_path_bergmann"])\
  .trigger(once=True)\
  .toTable("bergmann_binary_audio_with_ids")

# COMMAND ----------

df = spark.read.table("bergmann_binary_audio_with_ids")
display(df.limit(10))

# COMMAND ----------


