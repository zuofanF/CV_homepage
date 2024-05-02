'''
streamlitとは？ >>Webアプリを簡単に作れるアイブラリです。
上から下に実行順でウィジェットが表示される

我再次成功地解决了streamlit库的安装问题。

local URL和Network URL的区别？
>>>>local URL:用自己的本机访问。Network URL 是用于在同一网络中的其他设备上访问应用的网址。这里的 IP 地址 172.30.227.137 是分配给运行该服务设备的本地网络IP地址。

Syntax Highlight 语法高亮
'''

import streamlit as st
from PIL import Image

st.title('Zuofan\'s World')
st.caption('Motto｜The distance between dreams and reality is called ACTION.')

#textの表示
st.header('Bio')
st.text('Zuofan is a student focuses on Web Development and Machine Learning.\n'
        'A graduate student in Ritsumeikan University.')


#画像表示
image = Image.open('IMG_6730.jpg')
st.image(image, width=250)

#text box
name = st.text_input('名前')
print(name)

submit_btn = st.button('Send')
if submit_btn:
    st.text(f'ようこそ！{name}さん！')





# #動画表示
# st.header('Video')
# st.caption('Below is a scene of my hometown captured by a drone, located in the south of China.')
# video_file = open('view.MP4', 'rb')
# video_bytes = video_file.read()
# st.video(video_bytes)

# st.subheader('Code')
# st.text('Below, the code that how to train CNN Model.')

