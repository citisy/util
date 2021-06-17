import jieba
import pandas as pd
import wordcloud
import matplotlib.pyplot as plt


def draw(contents, stopwords=None):
    text = ''
    for c in contents:
        text += ' '.join(jieba.cut(c)) + ' '

    wc = wordcloud.WordCloud(font_path='C:/Windows/Fonts/STZHONGS.TTF',
                             stopwords=stopwords,
                             background_color='white',
                             random_state=80,
                             width=1200,
                             height=600)

    wc.generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('wordcloud.txt',
                     encoding='utf8',
                     index_col=False,
                     header=None,
                     quoting=3,
                     sep='\t')

    contents = df.loc[:, 8].values
    stopwords = []

    draw(contents, stopwords)