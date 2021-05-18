## A collection of spark utils

def spark_conn():
    """
    Easy local spark setup
    """
    from pyspark.sql import SparkSession
    from IPython.core.display import display as _display, HTML as _HTML
    #     sc_conf = SparkConf()
    #     ##Many more config options are available - set depending on your job!
    #     sc_conf.setAll([]).setAppName('STTest')
    #     .config(conf = sc_conf)
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    url = sc._jsc.sc().uiWebUrl().get()
    _display(_HTML(f'<a href = "{url}" target = "_blank">Click here to open Spark UI</a>'))

    return spark