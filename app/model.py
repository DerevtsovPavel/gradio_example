from math import exp
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

def predictor(model,df):
    def on_submit(bath,sqft_living,view,grade):

        if view>4 or view<0:
            raise gr.Error("Значение вида должно быть в диапазоне от 0 до 4")
        elif grade>13 or grade<0:
            raise gr.Error("Значение оценки должно быть в диапазоне от 0 до 13")
        
        y = model.predict([[bath,sqft_living,view,grade]])

        # Диаграмма рассеяния
        fig,ax = plt.subplots()

        ax.scatter(df['sqft_living'], df['price_ln'], color = 'blue', alpha=0.5)
        ax.scatter(sqft_living, y, color = 'red')

        ax.set_xlabel('Жилая площадь')
        ax.set_ylabel('Натуральный логарифм цены')

        ax.set_title('Диаграмма рассеяния')
        ax.legend()

        #Линия регрессии по площади
        fig2,ax2 = plt.subplots()
        ax2.scatter(df['sqft_living'],df['price_ln'],color='blue',alpha=0.5)
        ax2.scatter(sqft_living, y, color = 'red')

        min_sqft_living = df.loc[df['sqft_living'].idxmin()].drop('price_ln').values.tolist()
        max_sqft_living = df.loc[df['sqft_living'].idxmax()].drop('price_ln').values.tolist()

        y_min_pred = model.predict([min_sqft_living])[0]
        y_max_pred = model.predict([max_sqft_living])[0]

        ax2.axline([min_sqft_living[1],y_min_pred],[max_sqft_living[1],y_max_pred],color='red')

        ax2.set_xlabel('Жилая площадь')
        ax2.set_ylabel('Натуральный логарифм цены')


        #Линия регрессии по оценке
        fig3,ax3 = plt.subplots()
        ax3.set_xlim(0,df['grade'].max() + 1) #-1,+1 для отступа

        ax3.scatter(df['grade'],df['price_ln'],color='blue',alpha=0.5)
        ax3.scatter(grade, y, color = 'red')

        min_grade = df.loc[df['grade'].idxmin()].drop('price_ln').values.tolist()
        max_grade = df.loc[df['grade'].idxmax()].drop('price_ln').values.tolist()

        y_min_pred = model.predict([min_grade])[0]
        y_max_pred = model.predict([max_grade])[0]

        ax3.axline([min_grade[3],y_min_pred],[max_grade[3],y_max_pred],color='red')

        ax3.set_xlabel('Оценка')
        ax3.set_ylabel('Натуральный логарифм цены')

        return y[0],fig,fig2,fig3
    return on_submit
