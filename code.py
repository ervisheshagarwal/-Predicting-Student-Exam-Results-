# Students will use a small dataset about student study habits to:
import pandas as pd 
df = pd.read_csv("C:\Projects\-Predicting-Student-Exam-Results-\student_data.csv")
df.info()

# Apply Linear Regression to predict a student’s final exam score based on hours studied. 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x = df[["StudyTime"]]
y = df["FinalExamScore"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
model_lr  = LinearRegression()
model_lr.fit(x_train, y_train)

y_pred_train_lr = model_lr.predict(x_train)
y_pred_test_lr = model_lr.predict(x_test)

print("Training set predictions:", y_pred_train_lr)
print("Test set predictions:", y_pred_test_lr)

# Use Logistic Regression to predict whether a student passes or fails based on attendance rate and study time. 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x = df[["StudyTime", "AttendanceRate"]] 
y = df["Pass/Fail"]     

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

model_logistic_r = LogisticRegression()

model_logistic_r.fit(x_train, y_train)

y_pred_train_lr = model_logistic_r.predict(x_train)
y_pred_test_lr = model_logistic_r.predict(x_test)

print("Training set predictions:", y_pred_train_lr)
print("Test set predictions:", y_pred_test_lr)

# Try K-Means Clustering to group students based on their study time and attendance, then describe each group’s characteristics.

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

k = 3   # num of clusters

# Adding random state for reproducibility
model_Kmeans = KMeans(n_clusters=k, random_state=42)

columns_for_clustering = ['AttendanceRate', 'StudyTime']

# This line will now work because 'df' exists
df_for_clustering = df[columns_for_clustering]

# Fit the model to the data
model_Kmeans.fit(df_for_clustering)

# --- To see the results, let's visualize them ---

# Get the cluster labels for each data point
labels = model_Kmeans.labels_
# Get the coordinates of the cluster centers
centers = model_Kmeans.cluster_centers_

# Add the cluster labels to our original DataFrame for easy plotting
df['cluster'] = labels

# Plot the data points, coloring them by their assigned cluster
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['AttendanceRate'], df['StudyTime'], c=df['cluster'], cmap='viridis', marker='o')

# Plot the cluster centers
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')

plt.title('K-Means Clustering of StudyTime  and AttendanceRate')
plt.xlabel('AttendanceRate')
plt.ylabel('StudyTime ')
plt.legend(handles=scatter.legend_elements()[0], labels=['High Attendance & High Study', 'Low Attendance & Low Study', 'Moderate Attendance & Study'])
plt.show()

# Describe each cluster's characteristics
for i in range(k):
    group = df[df['cluster'] == i]
    print(f"\nCluster {i}:")
    print(group[columns_for_clustering].describe())
