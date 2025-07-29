## DriveGPT4  复现

DriveGPT4 只提供评估代码，未提供训练代码。

主代码包括 data_split.py、eval_cap.py、eval_control.py。官方 readme 如下:

```
1. Name the output of DriveGPT4 as ```DriveGPT4_output.json```.
2. Run ```python data_split.py``` to split eval set into easy, median and hard. The split method can be customized.
3. Run ```python eval_cap.py``` for caption evaluation.
4. Run ```python eval_control.py``` for control evaluation.
```

### data_split.py 

将数据分割为 easy, meidan, hard。注意到输出格式可知为从前两个问答对的中 model 回答（即奇数索引）中关键词来判定数据类型。

具体实现：
``` python
def find_word(word,json_data):
    if word in json_data['label'][1]['value'] or word in json_data['label'][3]['value']:
        return True
    else:
        return False
```
``` python
for d in data:
    if find_word('turn',d) :
        hard.append(d)
    elif  find_word('lane',d) or find_word('change',d) or find_word('accelerate',d) :
        median.append(d)
    else:
        easy.append(d)
```

### eval_cap.py

该文件包含两个函数：`evaluate_on_coco_caption` 和 `get_results` 。

`evaluate_on_coco_caption` 接受输出文件 `res_file` 和标签文件 `label_file`，并返回图像描述的评估指标。
``` python 
def evaluate_on_coco_caption(res_file, label_file, outfile='eval.json'):
     
    coco = COCO(label_file)       
    cocoRes = coco.loadRes(res_file)  
    cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')

    cocoEval.params['image_id'] = cocoRes.getImgIds()

    cocoEval.evaluate()
    result = cocoEval.eval
     
    if not outfile:
        print(result)
    else:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result  
```

`get_results` 接受参数 `name`、`mode`、`select`，`name` 表示 `mode` 取值 ``




### eval_control






