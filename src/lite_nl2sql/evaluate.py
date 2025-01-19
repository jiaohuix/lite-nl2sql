from lite_nl2sql.eval import start_evaluate

# Config the input datasets
data_folder = "dbgpt_hub_sql/data"

# Config evaluation parameters
evaluate_args =  {
            "input": "./dbgpt_hub_sql/output/pred_sql_rslora.sql",
            "gold": "./data/spider/dev_gold.sql",
            "db": "./data/spider/database",
            "table": "./data/spider/tables.json",
            "etype": "exec",
            "plug_value": True,
            "keep_distict": False,
            "progress_bar_for_each_datapoint": False,
            "natsql": False,
}

start_evaluate(evaluate_args)