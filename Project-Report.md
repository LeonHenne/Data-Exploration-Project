# Data Exploration Project Report

## Predicting the survivability of the Titanic passengers

### Erläuterung und Vorstellung des Datensatzes

Der in diesem Projekt untersuchte Datensatz basiert auf den
Passagierdaten der am 15. April 1912 versunkenen Titanic. Zu 891
Passagieren sind Informationen bezüglich deren Person, wie Name,
Geschlecht, Alter und Familienstatus, sowie deren Aufenthalt, z.B.
Passagiernummer, Schiffsklasse, Ticket, Fahrpreis, Einstiegsort,
Kabinennummer und schließlich ihres Überlebensstatus bekannt.

Die Daten zu den Informationen liegen dabei in folgender Art und Weise
vor:

**Person:**

\- Name: Gibt Auskunft über die Anrede, Vor-, Zweit- und Nachname der
Person im Format "Nachname, Anrede Vorname Nachname"

\- Geschlecht: Information zur biologischen Geschlechtszugehörigkeit
über Identifikation als "male" und "female"

\- Alter: Die Spalte "Age" gibt das Lebensalter in Jahren an. Dabei wird
das Alter von Passagieren jünger als 1 und von Passagieren mit
geschätztem Alter als Dezimalzahl angegeben (z.B. 0.66 bzw. 33.5)

\- Familienstatus: Zu dem Familienstatus befinden sich im Datensatz die
Spalten "SibSp" und "Parch", welche für die Anzahl von "Siblings &
Spouses" (Geschwistern und Ehegatten) und "Parents & Children" (Eltern
und Kindern) stehen.

**Aufenthalt**

\- PassagierID: Die PassagierID ist eine fortlaufende Zahl, die in
Einserschritten für jeden Passagiereintrag hochgezählt wird.

\- Schiffsklasse: Die Titanic war in drei Aufenthaltsklassen eingeteilt,
welche jeweils dem Ticketpreis entsprechend eine unterschiedliche
Reisequalität boten. Daraus kann angenommen werden, dass diese
Schiffsklassen auch die unterschiedlichen Sozio-ökonomischen
Gesellschaftsschichten widerspiegelten.

\- Ticket: Information zur Ticketnummer, welche hohe Unterschiede im
Darstellungsformat aufzeigt, da teilweise beliebig lange
Zahlenkombinationen mit Buchstaben und Sonderzeichen verwendet wurden.
Die Darstellungsformate werden im Datensatz nicht erläutert.

\- Fahrpreis: Fahrpreis als Dezimalzahl ohne Währungseinheit.

\- Einstiegsort: Information zum Einstiegsort Cherbourg, Queenstown oder
Southhampton, welche in den Daten mit "C", "Q" oder "S" abgekürzt wird.

\- Kabinennummer: Keine bis mehrere Angaben zur Kabinennummer (z.B. C85)

### Datenvorverarbeitung

Die Datenvorverarbeitung beinhaltet in diesem Projekt besonders die
Behandlung von fehlenden Werten und das Reduzieren des Datensatzes auf
für die Ziel relevante Informationen.\
Beim Erforschen des Datensatzes fällt besonders die Spalte der
Kabinennummer auf. Diese wäre ein interessantes Untersuchungsobjekt
unter der Annahme, dass es Kabinen gäbe, welche näher an Rettungsbooten
lägen, und so vorteilhaft für manchen Passagieren gewesen wären.
Allerdings weist die Spalte fehlende Werte in Höhe von 77,1 % auf,
wodurch das Spaltenobjekt sich nur unter hohem Datenverlust in das
Projekt eingliedern ließe.

Weiterhin weist auch die Spalte mit der Altersinformation einen
nennenswerten Teil von 19.87 % fehlender Werte auf. Da das bloße
Ersetzen der fehlenden Werte mit dem Wert 0 einen Einfluss auf die
Gesamtverteilung hätte, ist die Spalte mit dem durchschnittlichen Alter
aufgefüllt worden, um so auch die Zeilenanzahl des Datensatzes nicht zu
reduzieren.

Zusätzlich wurde der Datensatz in der Informationsvielfalt um die
Informationen des Namens, der Passagiernummer, des Einstiegsortes, der
Ticketnummer und des Ticketpreises reduziert, da diese keinen logischen
erklärbaren Zusammenhang mit der Überlebenschance aufweisen.

### Feature Engineering

Beim Erzeugen und Transformieren von Informationsmerkmalen werden bei
diesem Datensatz lediglich kleine Änderungen vorgenommen, da recht
einfache Merkmale vorliegen, und das Erzeugen neuer Merkmale im Modell
zu einer neuen Dimension führe. Eine höhere Anzahl an Dimensionen würde
dafür Sorge tragen, dass nächste Nachbarn nicht mehr wirklich nah
zueinander sind.

Die Geschlechtsinformation wird anders als die anderen relevanten
Informationen nicht numerisch ausgedrückt, wodurch hier eine
Transformation stattfinden muss, um Abstände zwischen den Punkten
kalkulieren zu können. Folglich wird „Female" durch 0 und „Male" durch 1
ersetzt.

Ein weiteres Thema kann je nach Algorithmus die Skalierung der
Informationsmerkmale sein. Besonders beim Klassifizieren der Datenpunkte
über die nächsten Nachbarn, spielt eine normalisierte Distanz eine
wichtige Rolle. So soll verhindert werden, dass Merkmale lediglich
aufgrund ihrer Abstandsverteilung einen höheren Einfluss auf die
Klassifikation nehmen. Aus diesem Grund werden die relevanten
Informationsmerkmale skaliert, indem der Mittelwert abgezogen und durch
die Standardabweichung dividiert wird.

### Teilen des Datensatzes in Trainings-, Validierungs- und Testdaten

Nachdem der Datensatz vollständig vorverarbeitet ist, wird dieser in
drei unterschiedliche Datensätze geteilt, um das Model auf nicht
gesehenen Daten zu validieren und zu testen. Über die integrierte
Funktion der sklearn Softwarebibliothek, wird den Trainingsdaten 534
Datenpunkte (60%), den Validierungsdaten 178 Datenpunkte (20%) und den
Testdaten 179 Datenpunkte (20%) der gesamten vorliegenden Daten
zugewiesen.

### Erläuterung des Machine Learning Algorithmus

Das im Projekt untersuchte Problem ist die richtige Identifizierung der
Passagiere als Überlebender oder nicht Überlebender. Zur Lösung dieses
Problems bedarf es daher einen Klassifizierungsalgorithmus, wie den
k-nearest neighbors Algorithmus (KNN Algorithmus), um den
Überlebensstatus der Passagiere richtig zuzuweisen. Der KNN Algorithmus
funktioniert nach dem Prinzip, dass die K nächsten Nachbarn, des zu
klassifizierenden Punktes betrachtet werden, um das in der Mehrheit
vorliegende Label dem neuen Punkt zuzuweisen. Hierfür muss jeder Punkt,
sowie jeder Abstand von jedem Punkt zu allen anderen Punkten berechnet
und gespeichert werden. Die Anzahl der betrachteten Nachbarn stellt bei
diesem Algorithmus einen Hyperparameter dar, welcher im späteren Kapitel
optimiert wird.

### Auswahl der Metriken

Dieses Projekt behandelt die Klassifikation der Passagiere zu ihrem
Überlebensstatus. Jegliche Metriken zur Evaluation des Algorithmus
basieren somit auf der korrekten oder inkorrekten Zuweisung des Labels 1
(überlebt) oder 0 (nicht überlebt). Daraus ergibt sich als Basis die
Konfusionsmatrix, welche die Informationen zu den wahren positiven,
wahren negativen, falschen positiven und falschen negativen enthält. Aus
dieser Matrix lassen sich weitere Metriken berechnen, wie z.B. die
Accuracy. Diese Metrik beschreibt den Anteil der korrekt klassifizierten
Punkte unter allen Punkten ($\frac{TP + TN}{TP + TN + FP + FN}$). Die
Accuracy wird im weiteren Verlauf als maßgebliche Metrik verwendet,
welche zu Optimieren ist.

### Training des KNN Algorithmus

Zu Beginn eines Laufs wird dem Algorithmus der Hyperparameter, die
Anzahl der zu betrachteten Nachbarn, übergeben. Anschließend kann die
fit-Methode das Model auf die Trainingsdaten der Merkmale und Label
anpassen und die Güte mittels der Validierungsdaten und Metriken
überprüft werden. Während des Trainings werden bei jedem Lauf
Informationen zu den verwendeten Datensätzen und den betrachteten
Nachbarn, sowie zu den Metriken über die Softwarebibliothek Mlflow
gespeichert.

### Optimierung der Hyperparameter

Zur Optimierung des Hyperparameters, wie viele Nachbarn bei der
Zuweisung eines Labels zu betrachten sind, wird der beschriebene
Trainings- und Validierungsprozess mehrfach durchgeführt. Dabei wird das
gesamte Intervall der natürlichen Zahlen von 1 bis 534 (Länge des
Trainingsdatensatzes) getestet.

Als Ergebnis dieser Optimierung ist in Mlflow folgende Grafik
entstanden: ![](media/image1.png){width="6.5in"
height="3.903041338582677in"}

Aus der Grafik lässt sich ableiten, dass der Parameter mit der besten
Accuracy auf den Validierungsdaten bei 21 betrachteten nächsten Nachbarn
liegt. Folglich wird das Model aus dem Lauf mit der RunID
„d51c7a19574240ecb8e03559d9b61a82" für die weitere Evaluation auf den
Testdaten verwendet.

### Evaluation auf den Testdaten ([DemoLink](https://colab.research.google.com/drive/1TrmB_XJCIk3vPSdiMGtkg1J3X44A8gvZ?usp=sharing))

Nachdem der optimale Hyperparameter bestimmt wurde, kann das
dazugehörige Model als Zip verpackt und als Github gist für ein anderes
Skript oder Notebook bereitgestellt werden. Beim Evaluieren des Models
auf den Testdaten, werden die Konfusionsmatrix und die Accuracy
berechnet und die Ergebnisse anhand von Visualisierungen dargestellt.
Das Ergebnis des Models auf Basis der Testdaten beträgt eine Genauigkeit
von 81% und folgende prozentuale Konfusionsmatrix:
$\frac{tpr:\ 52\%}{fpr:12\%}\frac{fnr:\ 6,7\%}{tnr:\ 29\%}$

### Schwachstellen und Verbesserungsmöglichkeiten im Algorithmus

Der K nearest neighbor Algorithmus bietet besonders in diesem Projekt
viele Vorteile, wie die geringe Komplexität und sein Verhalten als lazy
learning Algorithmus. Trotz dessen weist der Machine Learning
Algorithmus verschiedene Schwachstellen auf, welche anhand von
Verbesserungsmöglichkeiten kompensiert werden müssen.

Anhand seiner Funktionsweise bedarf der Algorithmus die Speicherung
jedes Punktes und die Berechnung aller möglicher Distanzen zwischen den
Punkten. Dies führt besonders bei skalierenden Datenmengen zu einem
erheblichen Speicherbedarf und einer langen Laufzeit. Ein Ansatz zur
Verbesserung dieser Problematik ist die Einführung einer
Vorabklassifikation, anhand von fixen Gitternetzen.[^1] Dabei werden die
Spannweiten der Dimensionsattribute ($D$) in beliebig gleich große
Intervalle ($I$), mit einer Anzahl von Datenpunkten ($N)$ geteilt,
woraus sich eine Gitternetzstruktur bildet. Anhand des Mittelpunktes
jedes Intervallbereiches kann anschließend mittels KNN Algorithmus die
Klasse des Intervallbereichs bestimmt werden. Die Komplexität dieser
Vorabklassifikation beträgt zwar $O(I^{D}\  \times \ N\ D)$, jedoch kann
anschließend ein neues Element mittels der Intervallklasse mit der
Komplexität $O(D)$ klassifiziert werden.

Eine weitere Schwachstelle des KNN Algorithmus, ist bekannt als „Fluch
der Dimensionalität". Mit jedem weiteren Attribut im Model erhöht sich
die Anzahl der Dimensionen, und folglich verringert sich die Dichte der
Datenpunkte. Dies ist erkennbar, an der minimalen Größe des Raums,
welcher alle nächsten Nachbarn enthält, welche sich wie folgt berechnet:
$\frac{k}{n}^{\frac{1}{d}}$. Bei erhöhter Distanz zwischen einem Punkt
und seinen nächsten Nachbarn, entkräftet sich die Annahme, dass Punkte
sich zu ihren nächsten Nachbarn ähneln.

Bei einer vorliegenden Datenstruktur mit vielen Attributen empfiehlt
sich daher einen Wechsel des Algorithmus in Betracht zu ziehen, oder
vorab eine Dimensionsreduktion mittels Principal Component Analysis
durchzuführen.

[^1]: Mestre, R. (2013). Improvements on the KNN classifier.
