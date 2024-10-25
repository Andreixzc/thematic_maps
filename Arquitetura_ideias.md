# Análise de Turbidez em Reservatórios

A princípio, a ideia é trabalhar com reservatórios pré-selecionados. Faremos o cadastro desses reservatórios de antemão e, em seguida, processaremos os dados, armazenando as análises em um banco de dados ou bucket. Assim, quando precisarmos exibir as informações ao usuário, poderemos simplesmente puxar os arquivos rasterizados `.TIF` e exibi-los no mapa do Earth Engine.

## Extração da Turbidez

Primeiramente, definiremos uma área do reservatório a ser analisada. Aplicaremos a máscara de água para identificar apenas os pixels correspondentes à água. 

Depois, realizaremos a análise pixel a pixel, utilizando o modelo carregado localmente (eventualmente na nuvem, como AWS Lambda e S3 Bucket). A partir dessa análise, geraremos os arquivos rasterizados com a turbidez calculada para cada pixel.

Para não exceder o limite do geemap, dividiremos a área de interesse em uma matriz de N x N "chunks" ou pedaços. Processaremos cada um individualmente e salvaremos localmente para utilização posterior. 

Após processar toda a área, mesclaremos todas as camadas e exibiremos o mapa de intensidade da turbidez.
