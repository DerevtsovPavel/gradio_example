import pickle
import gradio as gr
from model import *
import json
import pandas as pd

if __name__ == "__main__":
    lin_reg = pickle.load(open("../models/lin_reg_4_signs_sklear_ver=1.6.1__pickle_ver=4.0.pkl","rb"))
   
    with open('../models/model_param.json', 'r', encoding='UTF-8') as file:
        data = json.load(file)

    df = pd.read_csv('../small_df.csv')
    submit_handler = predictor(lin_reg,df)

    r2 = data['r2']
    cfs = data['coefs']
    free_cfs = data['free_coef']
    
    form = gr.Interface(fn = submit_handler,inputs = [gr.Number(label="Кол-во мокрых зон"),
                                             gr.Number(label = "Жилая площадь"),
                                             gr.Number(label="Оценка окружающего вида от 0 до 4"),
                                             gr.Number(label = "Общая оценка от 0 до 13")
                                             ],
                    outputs=[gr.Number(label="Натуральный логарифм предполагаемой цены дома"),
                             gr.Plot(label="Диаграмма рассеяния"),
                             gr.Plot(label="Линия регрессии по жилой площади"),
                             gr.Plot(label="Линия регрессии по оценке")],
                    article=f"Коэф-т детерминации:{r2}\n" + 
                    fr"$$ \hat y = {free_cfs} + {cfs[0]}x_1 + {cfs[1]}x_2 + {cfs[2]}x_3 + {cfs[3]}x_4 $$")# ур-е регрессии
    
    form.launch()