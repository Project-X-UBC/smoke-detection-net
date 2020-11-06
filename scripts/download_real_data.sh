#!/bin/bash
# Downloads all the data needed from OneDrive and organizes it.

data_path=../data/full
tools_path=../../src/tools

sudo apt install unzip
sudo apt install zip

cd $data_path

curl 'https://canadaeast1-mediap.svc.ms/transform/zip?cs=fFNQTw' -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' --compressed -H 'Content-Type: application/x-www-form-urlencoded' -H 'Origin: https://ubcca-my.sharepoint.com' -H 'DNT: 1' -H 'Connection: keep-alive' -H 'Cookie: spo_access_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNDY4NTYwMCIsImV4cCI6IjE2MDQ3MDcyMDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8bWVub25pQHN0dWRlbnQudWJjLmNhIiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDBlZDQwZjNiOUBsaXZlLmNvbSIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.ZmIybUJBemt5b1FHVVF2S3F1a0hxclFiSlFmSWFWQnRTMTkxa3NpZ2ZkZz0' -H 'Upgrade-Insecure-Requests: 1' --data-raw 'zipFileName=OneDrive_1_11-6-2020.zip&guid=8d21f484-1a86-4c5f-9c32-79a27c7d8a8f&provider=spo&files=%7B%22items%22%3A%5B%7B%22name%22%3A%22bigsmoke.zip%22%2C%22size%22%3A79239447%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4FWBM6R67QTQVC3XAQ3QXQS655L%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNDY4NTYwMCIsImV4cCI6IjE2MDQ3MDcyMDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8bWVub25pQHN0dWRlbnQudWJjLmNhIiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDBlZDQwZjNiOUBsaXZlLmNvbSIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.ZmIybUJBemt5b1FHVVF2S3F1a0hxclFiSlFmSWFWQnRTMTkxa3NpZ2ZkZz0%22%2C%22isFolder%22%3Afalse%7D%2C%7B%22name%22%3A%22frames_100.zip%22%2C%22size%22%3A1068425419%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4FGN5VR66LOJJHZNWTUDYDQXOFV%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNDY4NTYwMCIsImV4cCI6IjE2MDQ3MDcyMDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8bWVub25pQHN0dWRlbnQudWJjLmNhIiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDBlZDQwZjNiOUBsaXZlLmNvbSIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.ZmIybUJBemt5b1FHVVF2S3F1a0hxclFiSlFmSWFWQnRTMTkxa3NpZ2ZkZz0%22%2C%22isFolder%22%3Afalse%7D%2C%7B%22name%22%3A%22frames_200.zip%22%2C%22size%22%3A2100170177%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4DLNARL75UZVNEKXHZOWJVTNNW4%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNDY4NTYwMCIsImV4cCI6IjE2MDQ3MDcyMDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8bWVub25pQHN0dWRlbnQudWJjLmNhIiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDBlZDQwZjNiOUBsaXZlLmNvbSIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.ZmIybUJBemt5b1FHVVF2S3F1a0hxclFiSlFmSWFWQnRTMTkxa3NpZ2ZkZz0%22%2C%22isFolder%22%3Afalse%7D%2C%7B%22name%22%3A%22labels_1.json%22%2C%22size%22%3A263854146%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4DZ7CLT5LJA4NGJ23LQEITDPROJ%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNDY4NTYwMCIsImV4cCI6IjE2MDQ3MDcyMDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8bWVub25pQHN0dWRlbnQudWJjLmNhIiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDBlZDQwZjNiOUBsaXZlLmNvbSIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.ZmIybUJBemt5b1FHVVF2S3F1a0hxclFiSlFmSWFWQnRTMTkxa3NpZ2ZkZz0%22%2C%22isFolder%22%3Afalse%7D%2C%7B%22name%22%3A%22labels_2.json%22%2C%22size%22%3A276327588%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4EELMN3LLI47BEJY3ARGBHAQJDQ%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNDY4NTYwMCIsImV4cCI6IjE2MDQ3MDcyMDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8bWVub25pQHN0dWRlbnQudWJjLmNhIiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDBlZDQwZjNiOUBsaXZlLmNvbSIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.ZmIybUJBemt5b1FHVVF2S3F1a0hxclFiSlFmSWFWQnRTMTkxa3NpZ2ZkZz0%22%2C%22isFolder%22%3Afalse%7D%2C%7B%22name%22%3A%22labels_3.json%22%2C%22size%22%3A297116658%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4G5ABGLF2RVT5BKHYPWXMQUQCPW%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNDY4NTYwMCIsImV4cCI6IjE2MDQ3MDcyMDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8bWVub25pQHN0dWRlbnQudWJjLmNhIiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDBlZDQwZjNiOUBsaXZlLmNvbSIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.ZmIybUJBemt5b1FHVVF2S3F1a0hxclFiSlFmSWFWQnRTMTkxa3NpZ2ZkZz0%22%2C%22isFolder%22%3Afalse%7D%2C%7B%22name%22%3A%22labels_4.json%22%2C%22size%22%3A326221356%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4AP3YSCXNRBI5AJWKCMQJSNIVMM%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNDY4NTYwMCIsImV4cCI6IjE2MDQ3MDcyMDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8bWVub25pQHN0dWRlbnQudWJjLmNhIiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDBlZDQwZjNiOUBsaXZlLmNvbSIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.ZmIybUJBemt5b1FHVVF2S3F1a0hxclFiSlFmSWFWQnRTMTkxa3NpZ2ZkZz0%22%2C%22isFolder%22%3Afalse%7D%2C%7B%22name%22%3A%22labels_mankind_1.json%22%2C%22size%22%3A29028%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4AYCKZBUSFCNNAZBIJVCLH56M6Y%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNDY4NTYwMCIsImV4cCI6IjE2MDQ3MDcyMDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8bWVub25pQHN0dWRlbnQudWJjLmNhIiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDBlZDQwZjNiOUBsaXZlLmNvbSIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.ZmIybUJBemt5b1FHVVF2S3F1a0hxclFiSlFmSWFWQnRTMTkxa3NpZ2ZkZz0%22%2C%22isFolder%22%3Afalse%7D%2C%7B%22name%22%3A%22labels_mankind_2.json%22%2C%22size%22%3A35724%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4ACX7CI2AH66BGYVRRUXJ3FJFEE%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNDY4NTYwMCIsImV4cCI6IjE2MDQ3MDcyMDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8bWVub25pQHN0dWRlbnQudWJjLmNhIiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDBlZDQwZjNiOUBsaXZlLmNvbSIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.ZmIybUJBemt5b1FHVVF2S3F1a0hxclFiSlFmSWFWQnRTMTkxa3NpZ2ZkZz0%22%2C%22isFolder%22%3Afalse%7D%2C%7B%22name%22%3A%22labels_mankind_3.json%22%2C%22size%22%3A46884%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4B6VAY3XPPH6BFJPIRE5MPA5HTZ%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNDY4NTYwMCIsImV4cCI6IjE2MDQ3MDcyMDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8bWVub25pQHN0dWRlbnQudWJjLmNhIiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDBlZDQwZjNiOUBsaXZlLmNvbSIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.ZmIybUJBemt5b1FHVVF2S3F1a0hxclFiSlFmSWFWQnRTMTkxa3NpZ2ZkZz0%22%2C%22isFolder%22%3Afalse%7D%2C%7B%22name%22%3A%22labels_mankind_4.json%22%2C%22size%22%3A62508%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4G53MLP2RXWYVFLJOALJJIDJZUG%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNDY4NTYwMCIsImV4cCI6IjE2MDQ3MDcyMDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8bWVub25pQHN0dWRlbnQudWJjLmNhIiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDBlZDQwZjNiOUBsaXZlLmNvbSIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.ZmIybUJBemt5b1FHVVF2S3F1a0hxclFiSlFmSWFWQnRTMTkxa3NpZ2ZkZz0%22%2C%22isFolder%22%3Afalse%7D%2C%7B%22name%22%3A%22mankind.zip%22%2C%22size%22%3A25174449%2C%22docId%22%3A%22https%3A%2F%2Fubcca-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21a0W6yYOlHEq9bqvj4-iX9P1Ge72qAWlIiosm5BhtX_ZVWnMt14YgT4XV9mrFDJWk%2Fitems%2F01NYHGA4FFDUERZL7QMBGYNO3FDJFWPHJ6%3Fversion%3DPublished%26access_token%3DeyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdWJjY2EtbXkuc2hhcmVwb2ludC5jb21AMmZmZjA4YzktOTFkNC00ZmM4LWJiZGQtZGQ1OWI3NDE0ZGRiIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwibmJmIjoiMTYwNDY4NTYwMCIsImV4cCI6IjE2MDQ3MDcyMDAiLCJlbmRwb2ludHVybCI6IjNhYjVqOGN6UU0xdjFwZkNNTStYQTRlbTBjVnhBS29PWnlkZ0FJQmF2c2s9IiwiZW5kcG9pbnR1cmxMZW5ndGgiOiIxMTUiLCJpc2xvb3BiYWNrIjoiVHJ1ZSIsInZlciI6Imhhc2hlZHByb29mdG9rZW4iLCJzaXRlaWQiOiJZemxpWVRRMU5tSXRZVFU0TXkwMFlURmpMV0prTm1VdFlXSmxNMlV6WlRnNU4yWTAiLCJuYW1laWQiOiIwIy5mfG1lbWJlcnNoaXB8bWVub25pQHN0dWRlbnQudWJjLmNhIiwibmlpIjoibWljcm9zb2Z0LnNoYXJlcG9pbnQiLCJpc3VzZXIiOiJ0cnVlIiwiY2FjaGVrZXkiOiIwaC5mfG1lbWJlcnNoaXB8MTAwMzIwMDBlZDQwZjNiOUBsaXZlLmNvbSIsInR0IjoiMCIsInVzZVBlcnNpc3RlbnRDb29raWUiOiIyIn0.ZmIybUJBemt5b1FHVVF2S3F1a0hxclFiSlFmSWFWQnRTMTkxa3NpZ2ZkZz0%22%2C%22isFolder%22%3Afalse%7D%5D%7D&oAuthToken=' -o everything.zip

# Fix a corrupted file issue. Takes 10 min
sudo zip -FFv everything.zip --out everything_fixed.zip

sudo unzip everything_fixed.zip

sudo unzip frames_100.zip
sudo unzip frames_200.zip
sudo unzip bigsmoke.zip
sudo unzip mankind.zip

# rm everything.zip
# rm everything_fixed.zip

cd $tools_path

sudo python make_real_data_json.py
