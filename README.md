# Tesis Ingeniería Biomédica: Detección de cambios en mamografías y evolución de calcificaciones mamarias

## Descripción
Las estadísticas actuales sitúan el cáncer de mama como la principal causa de mortalidad debida a cáncer en mujeres. Por ello la detección precoz y cribado es esencial para disminuir el riesgo, mejorar el pronóstico y elevar la supervivencia. La OMS recomienda como método eficaz de cribado la realización de mamografía, debiendo ser anual para mujeres de edad media (40-50 años) en adelante.

El tejido mamario está compuesto por glándulas mamarias, conductos galactóforos y tejido de sostén (tejido mamario denso fibroglandular), y tejido graso (tejido mamario no denso). Todo el tejido mamario está vascularizado principalmente por vasos perforantes de la arteria y venas mamarias internas, situados a los lados del esternón. También recibe vascularización de los vasos torácicos laterales, rama de la arteria axilar. Otras arterias que aportan vascularización a la mama son los intercostales y toraco-acromiales. El líquido intersticial de la glándula mamaria es drenado mediante los vasos linfáticos de la mama a través de los linfáticos interlobulillares que confluyen formando el plexo linfático subareolar. Todos ellos drenan a los ganglios linfáticos, situados principalmente en la axila, aunque también puede estar en las proximidades de los vasos mamarios internos e incluso supraclaviculares. Este drenaje linfático tiene especial relevancia sobre todo en los tumores malignos, que usan los vasos linfáticos para propagar la enfermedad a distancia.

Las calcificaciones mamarias son un indicador de riesgo al momento de evaluar tumores en las mamografías. Las características que el mismo posee como distribución, morfología y composición son determinantes. Las mismas pueden ser benignas o malignas, produciéndose en general en áreas de necrosis las cuales proliferan células tumorales al usar el suministro sanguíneo que el área necrosada requiere dando como resultado una acidosis que provoca un microambiente para la acumulacion de calcio en los conductos.

La detección de cambios se entiende como el proceso de identificar diferencias en el estado de un objeto o fenómeno, observándolo en diferentes momentos. En el ámbito de la medicina, la detección de cambios es útil en el estudio de la evolución de una enfermedad o tumor. El objetivo principal de estas técnicas es identificar el conjunto de pıxeles que cambiaron “significativamente” entre una imagen y la otra. El proceso es complejo, ya que requiere distinguir los cambios que son relevantes de aquellos que no son de importancia, tales como ruido, diferente iluminación, cambios naturales debido al tiempo, etc.

Hoy en día, los cambios en las calcificaciones mamarias son evaluados subjetivamente mediante la comparación visual de dos mamografías por el profesional a cargo, estableciendo un criterio subjetivo basado en la experiencia del profesional. Este proyecto busca brindar al profesional una herramienta computacional que facilite la comparación para que pueda estudiar dicha evolución. Logrando de esta forma un diagnóstico apoyado por dicha herramienta para permitirle alertar tempranamente ante cambios en las mismas.

## 1. Install

You can use `Docker` to easily install all the needed packages and libraries. 

```bash
$ docker build -t tesis -f Dockerfile .
```

### Run Docker

```bash
$ docker run --rm --net host -it -v $(pwd):/home/app/src --workdir /home/app/src tesis bash
```