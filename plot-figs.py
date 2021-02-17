import pandas as pd
from matplotlib import pyplot as plt


# NAS에 올린 결과 csv 기준으로 작성
category = input('데이터 종류 입력 (pv / smp / load): ')             # pv / smp / load
dir = input('데이터 경로 입력 (NAS 업데이트 파일 기준, 절대경로): ')
start_idx = int(input('프로파일 플롯 시작 일자 (일 단위 입력, 숫자만, 걍 0으로 해도 됨): '))          # 프로파일 플롯 시작하는 날짜, 일 단위로 맞춰주세요 (ex) 2018.4.8부터 시작이면 7으로 인풋)
dataset = pd.read_csv('D:/GITHUB/python_projects/kepco-powerprice-prediction/data/예측데이터/smp_0813.csv', index_col=0)


##############################
true = dataset.iloc[:, 0]
pred = dataset.iloc[:, 1]

# true data profile
color = 'orange' if category=='pv' else 'purple' if category=='smp' else 'purple'
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(5,3.8))
plt.plot(true.values[start_idx*24:(start_idx+3)*24], color=color)          # color: pv='orange', load='purple', smp='green'
plt.xlim([0, 24*3])
plt.tight_layout()
plt.xticks(ticks=[i for i in range(0, 24*3, 24)],
           labels=['                          '+dataset.index[i][:10] for i in range(start_idx*24, 24*(3+start_idx), 24)])
plt.savefig(f'ppt_profile_{category}.png')
print('profile plot saved')


# prediction result
plt.rcParams.update({'font.size': 10})
plt.figure(figsize=(5,2.5))
plt.plot(true.values, alpha=0.7)
plt.plot(pred.values, color=color)                 # color: pv='orange', load='purple', smp='green'
plt.legend(['obs.', 'pred.'], loc='upper right')
plt.xlim([0, len(true)])
plt.tight_layout()
plt.xticks(ticks=[0, 24*31],
           labels=['                                                2018-04',
                   '                                                2018-05'])
plt.savefig(f'ppt_result_{category}.png')
print('result plot saved')