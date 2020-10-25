#!/bin/bash
# Downloads all the data needed from OneDrive and organizes it. Note that this could take up to 3 hours to download and split the videos

data_path=../data/full

sudo apt install unzip
sudo apt install zip

cd $data_path

# Download the videos. Takes 20 min
curl 'https://canadaeast1-mediap.svc.ms/transform/zip?cs=fFNQTw' \
  -H 'authority: canadaeast1-mediap.svc.ms' \
  -H 'cache-control: max-age=0' \
  -H 'sec-ch-ua: "Chromium";v="86", "\"Not\\A;Brand";v="99", "Google Chrome";v="86"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'origin: https://ubcca-my.sharepoint.com' \
  -H 'content-type: application/x-www-form-urlencoded' \
  -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Brave Chrome/83.0.4103.116 Safari/537.36' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
  -H 'sec-fetch-site: cross-site' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-dest: iframe' \
  -H 'accept-language: en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7,ja-JP;q=0.6,ja;q=0.5,zh-TW;q=0.4,zh;q=0.3' \
  --data-raw 'zipFileName=raw_data.zip&guid=57988989-4cc2-4406-a1c3-f5db7b8ec9d4&provider=spo&files=%7B%22items%22%3A%5B%7B%22name%22%3A%22raw_data%22%2C%22size%22%3A0%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4BTXS3QEWCA5BCLPKUWLTMRZ67N%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwMzU3MzIwMCIsImV4cCI6IjE2MDM1OTQ4MDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJzaWduaW5fc3RhdGUiOiJbXCJrbXNpXCJdIiwibmFtZWlkIjoiMCMuZnxtZW1iZXJzaGlwfG1lbm9uaUBzdHVkZW50LnViYy5jYSIsIm5paSI6Im1pY3Jvc29mdC5zaGFyZXBvaW50IiwiaXN1c2VyIjoidHJ1ZSIsImNhY2hla2V5IjoiMGguZnxtZW1iZXJzaGlwfDEwMDMyMDAwZWQ0MGYzYjlAbGl2ZS5jb20iLCJ0dCI6IjAiLCJ1c2VQZXJzaXN0ZW50Q29va2llIjoiMyJ9.TytNNGhndWdlMDZOQUxhalptV3R6YUJwYTUyYmR1NzNBWXFUazZLbmU5bz0%22%2C%22isFolder%22%3Atrue%7D%5D%7D&oAuthToken=' \
  --compressed -o raw_data.zip

# Fix a corrupted file issue. Takes 10 min
zip -FFv raw_data.zip --out raw_data_fixed.zip

unzip raw_data_fixed.zip

# Download labels.json
curl 'https://ubcca-my.sharepoint.com/personal/menoni_student_ubc_ca/_layouts/15/download.aspx?UniqueId=3fd66112%2D8f1a%2D46e4%2D88b7%2Dbe446a7981f8' \
  -H 'authority: ubcca-my.sharepoint.com' \
  -H 'sec-ch-ua: "Chromium";v="86", "\"Not\\A;Brand";v="99", "Google Chrome";v="86"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
  -H 'sec-fetch-site: same-origin' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-dest: iframe' \
  -H 'referer: https://ubcca-my.sharepoint.com/personal/menoni_student_ubc_ca/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmenoni%5Fstudent%5Fubc%5Fca%2FDocuments%2FProjectX' \
  -H 'accept-language: en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7,ja-JP;q=0.6,ja;q=0.5,zh-TW;q=0.4,zh;q=0.3' \
  -H 'cookie: MicrosoftApplicationsTelemetryDeviceId=750dc8d3-1c73-4cdb-befe-a0de85573aea; MicrosoftApplicationsTelemetryFirstLaunchTime=2020-10-24T22:59:48.848Z; rtFa=h7JyckFXoJnUYmQN0SPr4ChkB+piaxqw3qPOunefCiQmMkZGRjA4QzktOTFENC00RkM4LUJCREQtREQ1OUI3NDE0RERCK8BRGLapPCVMXK3hPOM21lKAgISriCuwgzSj2eo0A93xbe9lZHBHgJn7LmDn5dKWvMQJoZgP4kcNwQ6eldZ6+BZoNN+pJGGfnlB1g/fb6G44kArQSI5aHs90VJNGIoErps8k166YD6C3FTc0QzBLctMyQ1nFXiMlYRYzIgOvP5h1NIsNMZs/4C86cMQSo3IZbh+C6Wc6r/JNJaSgS9PgLuLNhLhELcj93PjuWDE4oNph7SQhgW/c7BbfoxZ8rZ0dHsDwzdiNcJmp+0XJ5I6dYfS3+23lPxLk9cbCdAM+A0zDbPuxXjTc5f4Sp4ec2/hOIDX2SGI/uXwFlsxRMyzxW0UAAAA=; CCSInfo=MTAvMjUvMjAyMCAxMTo0NDoxOSBQTVMu1rmOb5mz0RJlFOtcvh8IyIwCQV3wvCN3YIXzdEdw3ex8g0HbFVdEiFNRbi7zk7RGS9IgUrfmDzE5wZllI3fr7ikQE4ydIqmmErFRs5xg8uGvHl2D1C3FGCqQ53TqTASk2hBq/Q6oqS0CgumjYckwE78pdMOmBOlB7gU9DVXBkCCapm9MQwLj83LeLu8beCv5R90yLtLMw0TRUnVsKIlLOeECcmAgDd8PSeKpfHbMs4edKfuGHfgSUPyCrs5PjezZFwiE3+jHJ6leVCkKKrqJpedg04NeVfLCoan2jCTuTiTh8IsaqrBv1dSU+6eyREUEFN+VVwi4ZT3XgvQ8zhQWAAAA; cucg=0; FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjgsMGguZnxtZW1iZXJzaGlwfDEwMDMyMDAwZWQ0MGYzYjlAbGl2ZS5jb20sMCMuZnxtZW1iZXJzaGlwfG1lbm9uaUBzdHVkZW50LnViYy5jYSwxMzI0Nzg3NjM1MjAwMDAwMDAsMTMyNDcxMDkwOTkwMDAwMDAwLDEzMjQ4NDgzNjE1MzU1MDMzOSwxMDQuMjAwLjEzMi4xMywzLDJmZmYwOGM5LTkxZDQtNGZjOC1iYmRkLWRkNTliNzQxNGRkYiwsNjMxOTUwZmYtMzllNi00MjliLTk2ZTYtMTg1OTllNDM5MTYyLGQ1NDY4NjlmLWYwZDQtYjAwMC05ZDM2LWVmNTAzMmVlZDI1OSw0YTQ5MjQ2NS0xNjNjLTRjZjktYWM5MS0wMjU2NmQ1ZjljZjIsLDAsMTMyNDgwNTUyMTUzNTUwMzM5LDEzMjQ4MzEwODE1MzU1MDMzOSwsLGV5SjRiWE5mWTJNaU9pSmJYQ0pEVURGY0lsMGlmUT09LDI2NTA0Njc3NDM5OTk5OTk5OTksMTMyNDc4NzYzNjAwMDAwMDAwLDMyMWNjZjYyLTQ3MmUtNDdhZS05MDhlLTAyMTRhMzUzMTkwNSx1VlJVRlZUdXVxaHZBNzI0RFpSVUtNcFc0eFdSTkF2VTB6UlB5OHhFTXVldDUySTU3Qmpta1RjZ0tVc1RnRUR5WkVsejk3dnp3dE9xYS9lTGJJVG12Q1dubjRtZDRaVkU4Skd5VXZ5ckZlUDJBSURISDJSWVdjc0I0bFp2cnFOUmRMeEpJVkpNTTZzVUo0aFJCNlVJcHNaT3d0Z3dtWHB2TU9LM2JFOUtMRHlEdHZ2TDRJTjgybHAzZ24ybjJDZkRRdkNZYkdYWm5XKzNQc3RGK2JWU2hETVU1U1dZRXlRYTBxNjNnWHowMjQyNEtXV0d0NFYvbERXVHlmczJZSUhmWFNJS3pzNGpYMmg3MXhlMy9WdXVTa0NoeUlqbnlOVGlvTXZZNWthWHlIcWloa2czcGJzaW55MGtlY0dGbCtvOHhCVHFzYVNNU0E1YmYrU3lEejh6TUE9PTwvU1A+' \
  --compressed -o labels.json

cd ../../scripts

# Splits the videos into frames. Takes 1-2h depending on how many frames you want per video
python video_splitter.py

cd ../src/tools

python make_real_data_json.py
