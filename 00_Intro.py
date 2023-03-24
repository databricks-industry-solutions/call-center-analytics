# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC The goal of this accelerator is to demonstrate building a state-of-the-art transcription and content analytics pipeline at scale on Databricks. 
# MAGIC 
# MAGIC ## Business context
# MAGIC 
# MAGIC [*TBD: We can publish under Retail and expand on a retail use case, but the current dataset as suggested by Meng is best suited for public sector*]
# MAGIC 
# MAGIC ## Why use Databricks for this problem? 
# MAGIC * Databricks allows you to deploy open-source large language models in your own environment, directly integrated with the rest of your lakehouse ETL pipelines. In the context of this solution accelerator, we showcase how NLP tasks such as transcription (or automated voice recognition, ASR) and zero-shot topic modeling can be simply parallelized with Spark and becomes a segment of the overall call-processing streaming pipeline. 
# MAGIC * Hosting your own model may have benefits over using proprietary APIs for improved cost performance while avoiding vendor lock-in and data privacy concerns. 
# MAGIC * Your organization may finetune open-sourced models with internal data to cater to your specific use case. See how finetuning can be done on Databricks in this blog. [TODO Link Sean blog for finetuning here]
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Solution Outline
# MAGIC 
# MAGIC * In notebook 01, we setup a cloud storage location with audio (.mp3) files each representing individual calls. We use autoloader to incrementally and efficiently processes new data files as they arrive in cloud storage without any additional setup. We showcase how to transribe using an off-the-shelf [whisper transformer model](https://huggingface.co/bofenghuang/whisper-medium-cv11-german) from the HuggingFace Model Hub.
# MAGIC * In notebook 02, we perform zero-shot topic modeling to classify calls into known categories.

# COMMAND ----------

# MAGIC %md ##Create init script

# COMMAND ----------

# make folder to house init script
dbutils.fs.mkdirs('dbfs:/tutorials/LibriSpeech/')

# write init script
dbutils.fs.put(
  '/tutorials/LibriSpeech/install_ffmpeg.sh',
  '''
#!/bin/bash
apt install -y ffmpeg
''', 
  True
  )

# show script content
print(
  dbutils.fs.head('dbfs:/tutorials/LibriSpeech/install_ffmpeg.sh')
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Dataset
# MAGIC 
# MAGIC The sample dataset for this accelerator is a corpus of German emergency phone calls, recorded by Jörg Bergmann and placed into CHAT-CA format by Johannes Wagner.
# MAGIC 
# MAGIC **Citation information**
# MAGIC 
# MAGIC Bergmann, J. (1993). Alarmiertes verstehen: Kommunikation in Feuerwehrnotrufen. Wirklichkeit im Deutungsprozeß. In T. Jung & S. Müller-Doohm (Eds.), Verstehen und Methoden in den Kultur-und Sozialwissenschaften (pp. 283-328). Frankfurt aM: Suhrkamp.

# COMMAND ----------


