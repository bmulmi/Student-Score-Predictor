# Student-Score-Predictor
Data Analysis


Prepared certain hypotheses on a provided dataset (Cumulative.csv) of student's information and performance in courses and standardized tests.


Used Random Forest Classifier and Decision Tree Classifier from sklearn library to answer these hypotheses.


The fields in the file (Cumulative.csv) are comma-separated, but not column-aligned, and are as follows (listed along with the order in which they appear):

1. Identity Number - used only to identify records

Demographic information:
2. Sex (M/F)
3. Race encoded into a number
4. FirstGeneration (0 = no, 1 = yes)

Prior prepartion information:
5. SAT Reading score
6. SAT Math score
7. High school GPA

Treatment factors:
10. Major Type (1=CS, 2=Math, 3=Science, 4=Non-Science)
17. Instructor of Intro Course (encoded 1-9)
18. Instructor of Followup Course
19. Instructor of Fundamentals Course
20. Instructor of Systems Course

Outcome variables:
8. Semesters taken to graduate (provided only for those who have already graduated)
9. Cumulative GPA (provided only for those who have already graduated)
11. Grade in Intro Course (Range: 0 - 4.0)
12. Grade in Followup Course
13. Grade in Fundamentals Course
14. Grade in Systems Course
15. Grade in Software Course
16. Grade in Paradigms Course
