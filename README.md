# Mi-projet-Bio-Info

Rapport
Le changement climatique peut-il être observé dans les Alpes?


Partie 1:
Exploration des données
Quelles sont les dimensions de “meteo” ? 
	
	“meteo” est un fichier csv, avec 1294509 lignes et 58 colonnes.
	
Combien y a-t-il de postes météorologiques ?
	
	Au début et sans l'élimination de valeurs manquantes, il y a 142 postes différents.

Quelles colonnes allez-vous sélectionner pour notre étude ?
	
   	Ce fichier décrit les données sur les précipitations, température et le vent en                 Haute-Savoie entre 1950 et 2023.

 	Pour notre étude, on va éliminer toutes les valeurs qui font référence au Vent, et on va  garder toutes les valeurs qui décrivent la température et les précipitations.
Voici la liste des valeurs sélectionnées : 

'NUM_POSTE', 'NOM_USUEL', 'LAT', 'LON', 'ALTI', 'AAAAMMJJ', 'RR', 'QRR', 'TN', 'QTN', 'HTN', 'QHTN', 'TX', 'QTX', 'HTX', 'QHTX', 'TM', 'QTM', 'TNTXM', 'QTNTXM', 'TAMPLI', 'QTAMPLI', 'TNSOL', 'QTNSOL', 'TN50', 'QTN50', 'DG', 'QDG', 'DRR' et  'QDRR'

Contrôle de qualité:
	Toutes les les données reliés à la température et aux précipitations on une donnée associée, qui commence par Q, qui détermine la qualité des données.
	D’après la description des données, un 2 dans les colonnes de qualité représentent des valeurs douteuses. On va donc éliminer toutes les lignes où il y a des valeurs douteuses.
	Suite à l'élimination de ces valeurs on s’intéresse plus aux valeurs de qualité, donc on peut les éliminer de meteo_cleaned. 
	Il nous reste donc à remplacer ou éliminer les NaN qui restent dans notre Data Set. Le nombre de NaN (valeurs manquantes) et trop grand pour certaines colonnes, donc on va remplacer ces valeurs par la moyenne de la colonne, afin de garder le nombre maximal de données.

Combien de stations restent dans notre étude ? Où se situent-elles ?
	
	Après avoir fait notre “meteo_cleaned”, il reste 93 postes différents. Voici la liste des localisation:



Analyse des données

Quelles sont les tendances annuelles dans les données météorologiques depuis 1950 ? La température moyenne a-t-elle changé ? Est-ce qu'il y a plus ou moins de précipitations ?


Après avoir réalisé une régression linéaire pour estimer l’évolution des valeurs, on constate pour l’évolution de la température qu’il y a une hausse de 0.017 °C/an, et pour l’évolution des précipitations on constate qu’il n’y a presque pas de hausse.
Entre 1950 et 1980 on constate qu’il y a eu une diminution de la température (d’environ 8.7 °C à 7.2 °C ) et entre 1980 et 2023 on constate qu’il y a eu une diminution de la même (d’environ 7.2 °C à 10.2 °C en 2023 ). 
Les précipitations varient assez entre chaque année, mais on constate qu’au cours du temps, elles restent autour de 3.7 mm par jour en moyenne.


La régression est-elle pertinente pour la température ? Pour les précipitations ?

	En comparant les valeurs moyennes de la température avec la régression, elle semble être assez pertinente, cependant comme on a une descente puis une montée des températures , une régression linéaire semble ne pas être suffisamment précise. De plus, on constate que le nuage de points est assez éloigné de la droite. Pour les précipitations, on constate que la régression linéaire est pertinente, puisque même si le nuage de points est assez grand, la droite représente assez bien l’évolution des précipitations.

Quelle température fera-t-il en 2100 selon votre modèle ?

La température prédit pour l'année 2100 est de 10.43 °C.

Manipulez les données pour les regrouper mois par mois. Y a-t-il des tendances ? Correspondent-elles à vos connaissances ?

	On trouve bien qu’il y a des tendances selon les mois, surtout pour la température.
En été, il fait beaucoup plus chaud qu’en hiver (17.5 °C de différence entre Juillet et Janvier), ce qui est totalement normal.
	Pour les précipitations on trouve qu’il n’y a presque pas de différence entre été et hiver, mais c’est normal aussi puisque c’est un climat froid et il pleut assez pendant toute l’année.

Enfin si vous êtes à l’aise, étudiez en plus les épisodes de fortes chaleurs (températures > 28°C). Est-ce qu’il y en a plus souvent plus récemment?

	En effet, il y en a plus souvent plus récemment des journées avec plus de 28 °C, atteignant le nombre maximal de jours en 2022 avec 94 jours. Ceci est logique puisque une hausse de la température moyenne va impliquer souvent un hausse de la température maximale. En effet les années avec un nombre de journées avec plus de 28 °C ce sont les années autour des années 80, là où la température moyenne était la plus faible. 



Partie 2:

Avez-vous obtenu les mêmes résultats avec votre analyse de la partie 1?
	Oui

Regardez un peu les données. Quelles sont les dimensions des données historiques et des données de modélisation ?

	Les dimensions des données historiques sont de 39 lignes et 17 colonnes.
	Les dimensions des données de modélisation sont de 92 lignes et 17 colonnes.

Est-ce clair pour vous, quelle est la différence entre les données observées et historiques ?


Les données historiques permettent de comprendre les tendances à long terme du climat, tandis que les données observées fournissent des informations actuelles essentielles pour affiner les prévisions et ajuster les modèles en temps réel.



Discutez comment le changement climatique impacte les ressources en eau dans les régions montagneuses.
	
	Après l’analyse des données effectuée, on constate qu’historiquement, les débits cumulés de la rivière Arve ne cessent d’augmenter. Cependant les données de modélisation nous montre une future diminution de ces débit cumulés, surtout en été. En hiver, les données de modélisation montrent une augmentation des débits, mais qui est beaucoup plus faible que la variation espérée en été. Ceci on le voit aussi en regardant la courbe des débits cumulés par année. Cette courbe nous montre une future diminution importante des débits cumulés par an. En tenant en compte les analyses effectuées précédemment, on peut établir une corrélation entre l’augmentation de la température, probablement dûe au changement climatique, et la diminution espérée des débits d’eau cumulés. Une augmentation de la température générale implique une plus grande évaporation de l’eau, donc une diminution du débit d’eau cumulé.
