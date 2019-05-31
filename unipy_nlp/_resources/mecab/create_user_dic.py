"""Test Code Here.
"""

# %%

import os
import pandas as pd


udf_token = pd.DataFrame(
    [
        ['워라밸', 'T'],
        ['dt', 'F'],
        ['d/t', 'F'],
        ['dt총괄', 'T'],
        ['dt 총괄', 'T'],
        ['총괄', 'T'],
        ['digital transformation', 'T'],
        ['deep change', 'F'],
        ['deepchange', 'F'],
        ['happy hour', 'F'],
        ['해피 아워', 'F'],
        ['해피아워', 'F'],
        ['dt 총괄', 'T'],
        ['comm', 'T'],
        ['comm.', 'T'],
        ['mbwa', 'F'],
        ['캔미팅', 'T'],
        ['can meeting', 'T'],
        ['fu', 'T'],
        ['f/u', 'T'],
        ['모니터링', 'T'],
        ['dw', 'F'],
        ['d/w', 'F'],
        ['vwbe', 'F'],
        ['supex', 'F'],
        ['수펙스', 'F'],
        ['tm', 'T'],
        ['top', 'T'],
        ['탑', 'T'],
        ['its', 'F'],
        ['bottom up', 'T'],
        ['top down', 'T'],
        ['의사결정', 'T'],
        ['의사 결정', 'T'],
        ['self design', 'T'],
        ['self-design', 'T'],
        ['딜리버리', 'F'],
        ['delivery', 'F'],
        ['pt', 'F'],
        ['장표', 'F'],
        ['kpi', 'F'],
        ['hr', 'T'],
        ['h/r', 'T'],
        ['기업문화', 'F'],
        ['하이닉스', 'F'],
        ['이노베이션', 'T'],
        ['skt', 'F'],
        ['bm', 'T'],
        ['pm', 'T'],
        ['프로젝트', 'F'],
        ['pjt', 'F'],
        ['rm', 'T'],
        ['r/m', 'T'],
        ['culture', 'F'],
        ['cs', 'F'],
        ['c/s', 'F'],
        ['culture survey', 'F'],
        ['컬처 서베이', 'F'],
        ['컬쳐 서베이', 'F'],
        ['idp', 'F'],
        ['역량개발', 'T'],
        ['스탭', 'T'],
        ['스텝', 'T'],
        ['경영지원', 'T'],
        ['skcc', 'F'],
        ['sk cc', 'F'],
        ['sk cnc', 'F'],
        ['sk c&c', 'F'],
        ['ski', 'F'],
        ['이노베이션', 'T'],
        ['하이닉스', 'F'],
        ['텔레콤', 'T'],
        ['skh', 'F'],
        ['플래닛', 'T'],
        ['skp', 'F'],
        ['skc', 'F'],
        ['홀딩스', 'F'],
        ['sk 주식회사', 'F'],
        ['sk주식회사', 'F'],
        ['sk홀딩스', 'F'],
        ['sk주식회사 c&c', 'F'],
        ['sk 주식회사 c&c', 'F'],
        ['sk주식회사 cc', 'F'],
        ['sk주식회사cc', 'F'],
        ['self design', 'T'],
        ['selfdesign', 'T'],
        ['self-design', 'T'],
        ['경영협의회', 'F'],
        ['경영 협의회', 'F'],
        ['사업대표', 'F'],
        ['현장경영', 'T'],
        ['gtm', 'T'],
        ['vdi', 'F'],
        ['cloud-z', 'F'],
        ['cloudz', 'F'],
        ['cloud z', 'F'],
        ['cloud-edge', 'F'],
        ['cloudedge', 'F'],
        ['cloud edge', 'F'],
        ['클라우드', 'F'],
        ['사내 시스템', 'T'],
        ['사내시스템', 'T'],
        ['단기 성과', 'F'],
        ['단기성과', 'F'],
        ['watson', 'T'],
        ['왓슨', 'T'],
        ['유심포니', 'F'],
        ['선거운동', 'T'],
        ['연봉체계', 'F'],
        ['포괄 임금제', 'F'],
        ['포괄임금제', 'F'],
        ['장기 투자', 'F'],
        ['장기투자', 'F'],
        ['구성원 의사', 'F'],
        ['스마트워크', 'F'],
        ['스마트 워크', 'F'],
        ['smartwork', 'F'],
        ['여유시간', 'T'],
        ['여유 시간', 'T'],
        ['우리 회사', 'F'],
        ['우리회사', 'F'],
        ['외부 사이트', 'F'],
        ['외부사이트', 'F'],
        ['전사공지', 'F'],
        ['전사 공지', 'F'],
        ['점심시간', 'T'],
        ['점심 시간', 'T'],
        ['제조사업', 'T'],
        ['전략사업', 'T'],
        ['금융사업', 'T'],
        ['its사업', 'T'],
        ['its 사업', 'T'],
        ['물류서비스사업', 'T'],
        ['물류/서비스사업', 'T'],
        ['업무체계', 'F'],
        ['업무 체계', 'F'],
    ],
    columns=['word', 'last_yn'],
).drop_duplicates(subset='word')


udf_token['0'] = udf_token['word']
udf_token['1'] = 0
udf_token['2'] = 0
udf_token['3'] = 10
udf_token['4'] = 'NNG'
udf_token['5'] = '*'
udf_token['6'] = udf_token['last_yn']
udf_token['7'] = udf_token['word']
udf_token['8'] = '*'
udf_token['9'] = '*'
udf_token['10'] = '*'
udf_token['11'] = '*'
udf_token['12'] = '*'

udf_token_mecab = udf_token.loc[:, udf_token.columns.str.isnumeric()]


# """
#      0        1   2    3          4         5        6        7        8        9           10       11      12
# 표층형 (표현형태)	좌문맥ID  우문맥ID  출현비용     품사태그    의미부류   종성 유무    읽기      타입    첫번째품사	마지막 품사
# 서울              0      0        0         NNG       지명       T      서울       *        *           *         *       *
# 불태워졌	         0      0        0    VV+EM+VX+EP     *        T    불태워졌   inflected   VV          EP        *    불태우/VV/+어/EC/+지/VX/+었/EP/
# 해수욕장           0      0        0         NNG        *        T    해수욕장   Compound    *           *         *   해수/NNG/+욕/NNG/+장/NNG/*
# """


udf_token_mecab.to_csv(
    'mecab-ko-dic/user-dic/udf.csv',
    header=False,
    index=False,
)
