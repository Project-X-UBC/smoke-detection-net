#!/bin/bash
# Downloads all the data needed from OneDrive and organizes it. Note that this could take up to 3 hours to download and split the videos

data_path=../data/full

sudo apt install unzip
sudo apt install zip

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
  --data-raw 'zipFileName=raw_data.zip&guid=eb6dbf8a-3b4a-4f91-8d9f-9242f815aa5d&provider=spo&files=%7B%22items%22%3A%5B%7B%22name%22%3A%22raw_data%22%2C%22size%22%3A0%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4BTXS3QEWCA5BCLPKUWLTMRZ67N%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwMzQ4NjgwMCIsImV4cCI6IjE2MDM1MDg0MDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJzaWduaW5fc3RhdGUiOiJbXCJrbXNpXCJdIiwibmFtZWlkIjoiMCMuZnxtZW1iZXJzaGlwfG1lbm9uaUBzdHVkZW50LnViYy5jYSIsIm5paSI6Im1pY3Jvc29mdC5zaGFyZXBvaW50IiwiaXN1c2VyIjoidHJ1ZSIsImNhY2hla2V5IjoiMGguZnxtZW1iZXJzaGlwfDEwMDMyMDAwZWQ0MGYzYjlAbGl2ZS5jb20iLCJ0dCI6IjAiLCJ1c2VQZXJzaXN0ZW50Q29va2llIjoiMyJ9.NGtuUHZzYS9veVVzajFiZ1JtYnBydkRFYUNuZXdCL0ZaV2dGOEk4NVFSQT0%22%2C%22isFolder%22%3Atrue%7D%5D%7D&oAuthToken=' \
  --compressed -o $data_path/raw_data.zip

# Fix a corrupted file issue. Takes 10 min
zip -FFv raw_data.zip --out raw_data.zip

unzip raw_data.zip

# Download labels.json
curl 'https://ubcca-my.sharepoint.com/personal/menoni_student_ubc_ca/_layouts/15/download.aspx?UniqueId=3fd66112%2D8f1a%2D46e4%2D88b7%2Dbe446a7981f8' -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:81.0) Gecko/20100101 Firefox/81.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' --compressed -H 'Referer: https://ubcca-my.sharepoint.com/personal/menoni_student_ubc_ca/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmenoni%5Fstudent%5Fubc%5Fca%2FDocuments%2FProjectX' -H 'DNT: 1' -H 'Connection: keep-alive' -H 'Cookie: MicrosoftApplicationsTelemetryDeviceId=4bccf2b0-426c-4d8b-b9b8-c2775673377d; MicrosoftApplicationsTelemetryFirstLaunchTime=2020-10-23T00:42:53.739Z; rtFa=vCaocqeHJ+PHWyU0DqqAAfLaFTdkV5UgWki0EGlYer0mMkZGRjA4QzktOTFENC00RkM4LUJCREQtREQ1OUI3NDE0RERCZtiLiz8tbSMEbPSdMdRsCxHLx8pWNO4bWpwu36aAOxl36yaQoiB5QlRhjMXmA58CYL932VAryCQmB3fXgHv/wiXQidj1aRYz8Ugbzr7jexkGhKP/JgkJvqzK6at8coxyYcfdodi9Hxh59fij1lUwTGf60x/sK2/Ng3rf8rl+A9HZO/K6dAaWP4amRJ8ByOvtceRzdtiW+WJDrSj/gs6R67/ux090VifqXch8bRaGpxN70X8XVuIkvQBXdlLFdkOGpxF7tSvO0NoyT3oxGa29PADPfeFSEuXG0yrD9HwpDgi+aWyjN/S+lFzKwyY1+kA2vBnSFDwiekLVd3msusSMgkUAAAA=; FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjgsMGguZnxtZW1iZXJzaGlwfDEwMDMyMDAwZWQ0MGYzYjlAbGl2ZS5jb20sMCMuZnxtZW1iZXJzaGlwfG1lbm9uaUBzdHVkZW50LnViYy5jYSwxMzI0NzQ3MzkxNzAwMDAwMDAsMTMyNDcxMDkwOTkwMDAwMDAwLDEzMjQ4MzE5MzY5MjgwNDI2MywyNC44NC4yMTcuNjQsMywyZmZmMDhjOS05MWQ0LTRmYzgtYmJkZC1kZDU5Yjc0MTRkZGIsLGNjNDhjNDU5LWI4OGItNDQ4My1hOWNiLWIxY2Y4ZjgzYzFkNywwYmM3ODQ5Zi1hMDQ4LWIwMDAtNzI3Zi04MjQ5YTRlYTg3MzcsNTU1MTg2OWYtNjAzNC1iMDAwLTdhNGYtODFhNjRhNDRjOWEyLCwwLDEzMjQ3ODkwOTY5MjgwNDI2MywxMzI0ODE0NjU2OTI4MDQyNjMsLCxleUo0YlhOZlkyTWlPaUpiWENKRFVERmNJbDBpZlE9PSwyNjUwNDY3NzQzOTk5OTk5OTk5LDEzMjQ3NDczOTI2MDAwMDAwMCwzMjFjY2Y2Mi00NzJlLTQ3YWUtOTA4ZS0wMjE0YTM1MzE5MDUsa2MzTndnQjZ2Z05OTmRmZkdMOFB3dkYyOEdQa25Zb1dlL2VQMG9POE9QVkFEN1VqSnR0dTBsWVRhVDFTbUlHc3ZLbXE3NUxTeFZheEZrMDR0ZTF1d2NLcFFJbVNKMjQ5VXhvS1VWWFptMGorcFdvTW9haXloSEVFMUNMSS9kTjRkcmc5dmxFWHozRlVXcmpnU25qcTlSMDMzR0tibURRMW9TVUdxdElFQi81Y0haYzBacDlkUU1sb3VuWVdFZzBnQXNhZ1RUUm9QRUJ3ZnkxTVdGTGlBdjNwMkJrWWNPNW4vd3RDVGFzQnBkSGw4aUhjbEJPaVEyVFdJYVhwUE9uaTFpa2d0VWlZbk4yTVRWaHVNRzFKRXZiRDJhK0FMTlBDMHloWUNWckhGVkxONnRWckc0TC9HYjhEczFHekJFYnZvUy9FVG1lSVExYjRuazFsbkM3SGh3PT08L1NQPg==; cucg=0; odbn=1; WSS_FullScreenMode=false' -H 'Upgrade-Insecure-Requests: 1' -H 'TE: Trailers' -o $data_path/labels.json

cd ../../scripts

# Splits the videos into frames. Takes 1-2h depending on how many frames you want per video
python video_splitter.py

cd ../src/tools

python make_real_data_json.py