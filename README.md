# Smart Roulette: Lector Automàtic de Tirades

**Smart Roulette** és un sistema de visió per computador desenvolupat amb Python capaç d'identificar automàticament el número guanyador en una ruleta de casino a partir d'imatges estàtiques capturades d'un simulador digital. 

Aquest projecte s'ha desenvolupat com a pràctica per a l'assignatura de Visió per Computador (PSIV) de l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona (UAB).

## Característiques i Metodologia

El sistema processa les imatges mitjançant una primera etapa comuna de filtratge gaussià, emmascarament binari i l'aplicació de la **Transformada de Hough** per aïllar la posició de la bola blanca. A partir d'aquí, el projecte avalua i compara dues metodologies de classificació:

* **Mètode A (Geometria + KNN):** Calcula l'angle relatiu i invariant entre la bola i el número zero, classificant la tirada mitjançant l'algorisme de *K-Nearest Neighbors*. Aquest mètode ha demostrat ser altament robust, assolint una **precisió del 95%**.
* **Mètode B (Alineació + OCR):** Realitza una rotació compensatòria de l'escena per redreçar la casella guanyadora i n'extreu el text utilitzant un motor d'Intel·ligència Artificial (EasyOCR). Degut a les distorsions visuals, assoleix una **precisió del 75%**.

## Estructura del Projecte

* `main.py` - Controlador principal. Genera la màscara comuna i permet executar els diferents pipelines mitjançant un menú interactiu.
* `detectar_0.py` - Lògica del Mètode A (Extracció d'angles i model KNN).
* `ocr_detector.py` - Lògica del Mètode B (Retall de *patches* i lectura de caràcters amb EasyOCR).
* `ball_detector.py` - Funcions compartides per a la localització de primitives geomètriques.
* `preprocessing.py` - Eines generals de tractament i augment de dades.
* `requirements.txt` - Llistat de dependències necessàries per executar el projecte.

## Instal·lació

1. Assegura't de tenir **Python 3.x** instal·lat al teu sistema.
2. Clona aquest repositori o descarrega'n els fitxers.
3. Obre un terminal a la carpeta arrel del projecte i instal·la les llibreries necessàries executant:

```bash
pip install -r requirements.txt
