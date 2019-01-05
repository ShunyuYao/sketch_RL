import xml.dom.minidom
import glob
import os
import pandas as pd


def read_view(xml_path):
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    element = root.getElementsByTagName('track')[0]
    view_num = element.getAttribute('view')

    return view_num

def read_Emotion(xml_path):
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    element = root.getElementsByTagName('Metatag')[1]
    view_num = element.getAttribute('Value')

    return view_num

def parse_view(src_path):
    src_basenames = []
    views = []
    src_list = sorted(glob.glob(src_path + '/*/*[!a]*.xml'))  # error
    for i, src in enumerate(src_list):
        # if i > 10:
        #     break
        print(src)
        if 'sesson' in src:
            src_basename = os.path.basename(src)
            src_basename = src_basename.split('.')[0]
            src_basenames.append(src_basename)
        elif 'sesson' in src:
            views.append(read_view(src))
    return src_basenames, views

def parse_Emotion(src_path):
    src_basenames = []
    emotions = []
    src_list = sorted(glob.glob(src_path + '/*/S*.xml'))
    for i, src in enumerate(src_list):
        # if i > 10:
        #     break
        if 'oao' not in src:
            src_basename = os.path.basename(src)
            src_basename = src_basename.split('.')[0]
            src_basenames.append(src_basename)
            emotions.append(int(read_Emotion(src)))
        else: continue

    return src_basenames, emotions

if __name__ == '__main__':
    src_path = '/media/yaosy/办公/research300/videoSegData/mmi/mmi-facial-expression-database_download_2018-12-29_13_28_54/Sessions'
    # src_basenames, views = parse_view(src_path)
    src_basenames, emotions = parse_Emotion(src_path)
    emotion_df = pd.DataFrame({'name': src_basenames, 'emotion_id': emotions})
    print(emotion_df)
    emotion_df.to_csv('../dataset/emotion_id.csv',index=False)
    #print(views)
