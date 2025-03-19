import streamlit as st
import pandas as pd
import numpy as np
import pulp

def create_optimization_model(instructors, students, aircraft_count):
    # Create the optimization model
    model = pulp.LpProblem("Flight_Scheduling_Optimization", pulp.LpMaximize)
    
    # Create sets for indexing
    instructor_ids = list(range(len(instructors)))
    student_ids = list(range(len(students)))
    course_ids = [1, 2, 3]
    
    # Create decision variables - binary for assignment of instructor i to student j
    x = {}
    for i in instructor_ids:
        for j in student_ids:
            x[i, j] = pulp.LpVariable(f"assign_instructor_{i}_to_student_{j}", cat='Binary')
    
    # Objective function: Maximize the number of assigned students
    model += pulp.lpSum(x[i, j] for i in instructor_ids for j in student_ids)
    
    # Constraint 1: Each instructor can teach at most one student per day
    for i in instructor_ids:
        model += pulp.lpSum(x[i, j] for j in student_ids) <= 1
    
    # Constraint 2: Each student can be assigned to at most one instructor
    for j in student_ids:
        model += pulp.lpSum(x[i, j] for i in instructor_ids) <= 1
    
    # Constraint 3: Instructors can only teach courses they are qualified for
    for i in instructor_ids:
        for j in student_ids:
            for course in course_ids:
                # If instructor is not qualified for the course and student needs this course, they can't be assigned
                if course not in instructors[i]['qualifications'] and course == students[j]['course']:
                    model += x[i, j] == 0
    
    # Constraint 4: Limited by aircraft availability
    model += pulp.lpSum(x[i, j] for i in instructor_ids for j in student_ids) <= aircraft_count
    
    return model, x

def solve_and_display_results(model, x, instructors, students):
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Create results dataframe
    results = []
    assigned_students = 0
    
    for i in range(len(instructors)):
        for j in range(len(students)):
            if pulp.value(x[i, j]) == 1:
                assigned_students += 1
                results.append({
                    "Instructor ID": i + 1,
                    "Instructor Qualifications": str(instructors[i]['qualifications']),
                    "Student ID": j + 1,
                    "Student Course": students[j]['course']
                })
    
    total_students = len(students)
    unassigned_students = total_students - assigned_students
    
    return pd.DataFrame(results), assigned_students, unassigned_students

# Streamlit app
st.title("Flight Instructor Scheduling Optimization")

st.header("Input Parameters")

# Aircraft information
aircraft_count = st.number_input("Number of Available Aircraft", min_value=1, value=3)

# Instructor information 
col1, col2 = st.columns(2)
with col1:
    instructor_count = st.number_input("Number of Instructors", min_value=1, value=3)

# Container for instructor qualifications
instructors = []
with st.expander("Set Instructor Qualifications", expanded=True):
    for i in range(instructor_count):
        st.subheader(f"Instructor {i+1}")
        col1, col2, col3 = st.columns(3)
        with col1:
            qual1 = st.checkbox(f"Course 1", value=True, key=f"instr_{i}_course_1")
        with col2:
            qual2 = st.checkbox(f"Course 2", value=i > 0, key=f"instr_{i}_course_2")
        with col3:
            qual3 = st.checkbox(f"Course 3", value=i > 1, key=f"instr_{i}_course_3")
        
        qualifications = []
        if qual1:
            qualifications.append(1)
        if qual2:
            qualifications.append(2)
        if qual3:
            qualifications.append(3)
            
        instructors.append({"id": i+1, "qualifications": qualifications})
        st.write(f"Qualified for courses: {qualifications}")
        st.divider()

# Student information
student_count = st.number_input("Number of Students", min_value=1, value=5)

# Container for student courses
students = []
with st.expander("Set Student Courses", expanded=True):
    for i in range(student_count):
        st.subheader(f"Student {i+1}")
        course = st.radio(
            f"Course Selection for Student {i+1}",
            [1, 2, 3],
            index=i % 3,
            horizontal=True,
            key=f"student_{i}_course"
        )
        students.append({"id": i+1, "course": course})
        st.write(f"Enrolled in course: {course}")
        st.divider()

# Run optimization
if st.button("Run Optimization"):
    st.header("Optimization Results")
    
    # Display inputs summary
    st.subheader("Input Summary")
    st.write(f"Number of instructors: {instructor_count}")
    st.write(f"Number of students: {student_count}")
    st.write(f"Number of aircraft: {aircraft_count}")
    
    # Create and solve the model
    model, x = create_optimization_model(instructors, students, aircraft_count)
    results_df, assigned_students, unassigned_students = solve_and_display_results(model, x, instructors, students)
    
    # Display optimization status
    st.subheader("Schedule Results")
    if assigned_students > 0:
        st.success(f"Optimization completed! {assigned_students} students assigned.")
        st.warning(f"{unassigned_students} students could not be scheduled.")
        st.dataframe(results_df)
        
        # Calculate utilization metrics
        instructor_utilization = (assigned_students / instructor_count) * 100
        aircraft_utilization = (assigned_students / aircraft_count) * 100
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Instructor Utilization", f"{instructor_utilization:.1f}%")
        with col2:
            st.metric("Aircraft Utilization", f"{aircraft_utilization:.1f}%")
            
        # Display unassigned students if any
        if unassigned_students > 0:
            st.subheader("Unassigned Students")
            assigned_student_ids = results_df["Student ID"].tolist() if not results_df.empty else []
            unassigned_df = []
            
            for j in range(len(students)):
                if j+1 not in assigned_student_ids:
                    unassigned_df.append({
                        "Student ID": j+1,
                        "Course": students[j]['course']
                    })
            
            st.dataframe(pd.DataFrame(unassigned_df))
    else:
        st.error("No students could be assigned. Check instructor qualifications and constraints.")

# Add explanation of the model
with st.expander("About this Optimization Model"):
    st.markdown("""
    ### How the Model Works
    
    This application uses linear programming to optimize the daily scheduling of flight instructors, students, and aircraft.
    
    #### Objective:
    - Maximize the number of students assigned to instructors
    
    #### Constraints:
    1. Each instructor can teach at most one student per day
    2. Each student can be assigned to at most one instructor
    3. Instructors can only teach courses they are qualified for
    4. Total number of assignments is limited by aircraft availability
    
    #### Under-allocation Reduction:
    The model prioritizes matching as many students as possible while respecting all constraints,
    effectively minimizing the number of unassigned students.
    """)