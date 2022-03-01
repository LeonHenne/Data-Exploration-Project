#### Leon Henne, 22.02.2022
# Data Exploration Project
## Predicting the survivability of the Titanic passengers

### **Erläuterung und Vorstellung des Datensatzes**
Der in diesem Projekt untersuchte Datensatz basiert auf den Passagierdaten der am 15. April 1912 versunkenen Titanic. Zu 891 Passagieren sind Informationen bezüglich deren Person, wie Name, Geschlecht, Alter und Familienstatus, sowie deren Aufenthalt, z.B. Passagiernummer, Schiffsklasse, Ticket, Fahrpreis, Einstiegsort, Kabinennummer und schließlich ihres Überlebensstatus bekannt.

Die Daten zu den Informationen liegen dabei in folgender Art und Weise vor:

**Person:**
- Name: Gibt Auskunft über die Anrede, Vor-, Zweit- und Nachname der Person im Format "Nachnamme, Anrede Vorname Nachname"
- Geschlecht: Information zur biologischen Geschlechtszugehörigkeit über Identifikation als "male" und "female"
- Alter: Die Spalte "Age" gibt das Lebensalter in Jahren an. Dabei wird das Alter von Passagieren jünger als 1 und von Passagieren mit geschätztem Alter als Dezimalzahl angegeben (z.B. 0.66 bzw. 33.5)
- Familienstatus: Zu dem Familienstatus befinden sich im Datensatz die Spalten "SibSp" und "Parch", welche für die Anzahl von "Siblings & Spouses" (Geschwistern und Ehegatten) und "Parents & Children" (Eltern und Kindern) stehen.

**Aufenthalt**
- PassagierID: Die PassagierID ist eine fortlaufende Zahl die in Einserschritten für jeden Passagiereintrag hochgezählt wird.
- Schiffsklasse: Die Titanic war in drei Aufenthaltsklassen eingeteilt, welche jeweils dem Ticketpreis entsprechend eine unterschiedliche Reisequalität boten. Daraus kann angenommen werden, dass diese Schiffsklassen auch die unterschiedlichen Sozio-ökonomischen Gesellschaftsschichten widerspiegelten.
- Ticket: Information zur Ticketnummer, welche hohe Unterschiede im Darstellungsformat aufzeigt, da teilweise beliebig lange Zahlenkombinationen mit Buchstaben und Sonderzeichen verwendet wurden. Die Darstellungsformate werden im Datensatz nicht erläutert.
- Fahrpreis: Fahrpreis als Dezimalzahl ohne Währungseinheit auf.
- Einstiegsort: Information zum Einstiegsort Cherbourg, Queenstown oder Southhampton, welche in den Daten mit "C", "Q" oder "S" abgekürzt wird.
- Kabinennummer: Keine bis mehrere Angaben zur Kabinennummer (z.B. C85)

### **Datenvorverarbeitung**
Die Datenvorverarbeitung beinhaltet in diesem Projekt besonders die Behandlung von fehlenden Werten und das Reduzieren des Datensatzes auf für die Ziel relevante Informationen.

Beim Erforschen des Datensatzes fällt besonders die Spalte der Kabinennummer auf. Diese wäre ein interessantes Untersuchungsobjekt unter der Annahme, dass es Kabinen gäbe, welche näher an Rettungsbooten lägen, und so vorteilhaft für manchen Passagieren gewesen wären. Allerdings weißt die Spalte fehlende Werte in Höhe von 77,1 % auf, wodurch das Spaltenobjekt sich nur unter extrem hohem Datenverlust in das Projekt eingliedern ließe.

Weiterhin weißt auch die Spalte mit der Altersinformation einen nennenswerten Teil von 19.87 % fehlender Werte auf. Da das bloße Ersetzen der fehlenden Werte mit dem Wert 0 einen Einfluss auf die Gesamtverteilung hätte, ist die Spalte mit dem durschnittlichen Alter aufgefüllt worden, um so auch die Zeilenanzahl des Datensatzes nicht zu reduzieren.

Zusätzlich wurde der Datensatz in der Informationsvielfalt um die Informationen des Names, der Passagiernummer, des Einstiegsortes, der Ticketnummer und des Ticketpreises reduziert, da diese keinen logischen Zusammenhang mit der Überlebenschance aufweisen.

### **Feature Engineering**
- Age Spalte in Range zwischen 0 und 1 bringen ? (/ 100) ohne: 0.7415730337078652 mit : 0.7247191011235955
- Sex Spalte in 0 und 1 um sie zu Integern zu konvertieren und in einen vergleichbaren Raum zu projezieren
- Summe der Angehörigen berechnen (Parch + SibSp) ? in Bereich zwischen 0 und 1 bringen ? --> Macht kein Sinn da es lediglich den Informationsgehalt des Datensatzes reduziert
### **Teilen in Trainings- und Testdaten**

### **Erläuterung des Machine Learning Algorithmus**

### **Auswahl der Metriken**

### **Traingsergebnisse des Algorithmus**

### **Hyperparmametertuning**

### **Evaluation auf den Testdaten**

### **Schwachstellen und Verbesserungsmöglichkeiten im Algorithmus**
- Rechenzeit und Speicherbedarf (Jede Distanz von und zu jedem Punkt)