# Como rodar

Primeiramente, é preciso ter um projeto no Google Cloud que usa o Earth Engine. Após isso, precisamos autenticar o Earth Engine. Para isso, basta rodar no terminal:

```bash
earthengine authenticate
earthengine set_project nome-do-projeto

python -m venv nome_do_ambiente
nome_do_ambiente\Scripts\activate

pip install -r requirements.txt

```
