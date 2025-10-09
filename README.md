# 🚀 VISAO-COMPUTACIONAL_ERGOVIEW

**VISAO-COMPUTACIONAL_ERGOVIEW** é um projeto de visão computacional que permite [descrição breve do projeto]. Ele oferece [funcionalidades principais], utilizando [tecnologias principais, como OpenCV, TensorFlow, etc.].

---

## 📌 **Índice**
- [🔧 Instalação](#-instalação)
- [🚀 Como Usar](#-como-usar)
- [📂 Estrutura do Projeto](#-estrutura-do-projeto)
- [📜 Licença](#-licença)


---

## 🔧 **Instalação**

Para instalar e rodar o projeto localmente, siga estes passos:

### **1️⃣ Clone o Repositório**
```sh
git clone https://github.com/SeuUsuario/VISAO-COMPUTACIONAL_ERGOVIEW.git
cd VISAO-COMPUTACIONAL_ERGOVIEW
```

### **2️⃣ Execute o Script de Configuração**
```sh
setup.sh
```
### ** Caso esteja utilizando a versão integrada com a API: **
Primeiramente você deverá inicializar o container do Minio Storage por meio do arquivo docker-compose.yml localizado na raiz do projeto:
```
docker compose up --build -d
```
Para pausar a execução do container voce pode executar o seguinte comando:
```
docker compose down
```
Por padrão, o minio utiliza as portas 9000 e 9001 para o backend(API) e a interface frontend da aplicação respectivamente. Caso queira alterar a porta, você pode alterar o campo "ports" no arquivo docker-compose.yml:
```
    ports:
      - "Altere o valor que vem antes dos dois pontos:9000"
      - "para o valor da porta que deseja utilizar:9001"
```
Caso altere a porta da aplicação do minio, você deverá atualizar o arquivo de configuração "appsettings.json" localizado dentro da pasta src/ergoview. Este arquivo possui algumas configurações as quais o sistema utiliza ao iniciar.
```
    "Minio":{
        "baseUrl":"localhost:9000", -> se alterar a porta do minio, altere este campo
        "acess_key":"HGIjGxNBbi5EDSW9uB2I", -> este campo é o user do minio
        "secret_key":"hep38nO6raztiV2YzGktZLzWHjRwZPafkCD33cnc", -> este campo é a senha do usuário
        "bucket_name":"intelbras" -> este é o nome do bucket onde ficará armazenada os frames
    }

    #OBS: Voce pode utilizar os valores dos campos acess_key e secret_key para fazer login na interface
    frontend do minio na porta 9001.
```
Após finalizar as etapas de configuração, instale as dependências do projeto:
```
pip install -r requirements.txt
```
Em seguida execute o projeto:
```
python ./src/ergoview/main.py
```
### **🚀 Como Usar**
A api está configurada para subir na seguinte url: http://localhost:5000. A mesma possui dois endpoints cadastrados sendo eles:
```
- http://localhost:5000/api/start_record
- http://localhost:5000/api/calibration
```
Você pode utilizar uma ferramenta para testar os endpoints como Insomnia ou Postman. O primeiro endpoint é responsável pela captura de frames e processamento do rula, ao chamar eles os seguintes campos devem ser enviados no corpo da requisição:
```
{
	"id_monitoramento":"1", -> Id do monitoramento gerado pelo backend
	"index_cam":"0", -> Index da camera a qual será utilizada, pode ser 0, 1 ou 7 para utilizar ambas as cãmeras.
	"time":"10" -> tempo máximo de captura, aceita valores entre 60 segundo e 300, 1 a 5 minutos respectivamente
}
```
O segundo endpoint é responsável por retornar a lista de pontos referentes a calibração da câmera, nenhum parãmetro é necessário no corpo da requisição.

Caso queira visualizar os frames capturados pela durante a operação, é necessário acessar o bucket do minio e buscar pela pasta nomeada de acordo com o id_monitaramento, ela irá conter todas os frames referentes aquela sessão.
### **📂 Estrutura do Projeto**
VISAO-COMPUTACIONAL_ERGOVIEW/

    │── src/
    │── tests/              # Testes
    │── docs/               # Documentação
    │── README.md           # Instruções do projeto
    │── requirements.txt    # Dependências do projeto
    │── .github/            # Workflows do GitHub Actions
    │── LICENSE             # Licença do projeto

### **📜 Licença**
Este projeto é licenciado sob a licença MIT.
