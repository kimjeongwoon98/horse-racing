0.개요
반디집을 통해 zip 파일에 대한 압축해제를 수행해야 한다.   

1.데이터 전처리  
csv 데이터 행의 단위는 "track_id", "race_date", "race_number", "program_number"으로 형성된다.   
즉, 4개의 columns에 대해 groupby를 적용하여, unique row of data를 생성해야 한다. trakus_index가 시간에 대한 정보를 담고 있기 때문에, trakus_index를 통해 데이터를 정렬해야 한다.  
csv 파일 nyra_2019_complete_without_jockey.csv는 정렬을 수행했다.  

2.데이터 특성  
2.1. (위도, 경도, 시간)으로 기록된, 경마 자취 데이터는 총 14,915개 있으며, 각 케이스는 200-300개의 timestamp으로 구성되어 있다.
