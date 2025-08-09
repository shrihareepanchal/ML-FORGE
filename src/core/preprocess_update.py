def preprocess_data():
    if st.session_state.df is None:
        if os.path.exists("saved_data.csv"):
            st.session_state.df = pd.read_csv("saved_data.csv")
            st.info("Loaded saved data. Please proceed with preprocessing.")
        else:
            st.warning("Please upload data in the 'Data Loading' section first.")
            st.image(
                "no_data_image.png", width=400
            )  # Display image if no data, adjust path if needed
            return  # Exit preprocessing if no data is loaded

    st.subheader("📊 Data Preprocessing")

    # 1️⃣ Display Dataset Shape
    st.write(
        f"**Total Rows:** {st.session_state.df.shape[0]}, **Total Columns:** {st.session_state.df.shape[1]}"
    )

    # 2️⃣ Remove Duplicates
    duplicates = st.session_state.df.duplicated().sum()
    if duplicates > 0:
        st.session_state.df = st.session_state.df.drop_duplicates()
        st.write(f"✅ Removed {duplicates} duplicate rows.")
    else:
        st.write("✅ No duplicate rows found.")

    # 3️⃣ Check and Remove Null Values
    st.subheader("🔍 Missing Values Handling")

    # Count initial null values
    null_counts = st.session_state.df.isnull().sum()
    st.write("**Initial Null Values Count per Column:**")
    st.write(null_counts[null_counts > 0])

    # User-defined threshold for removing rows
    row_threshold = st.number_input(
        "Enter max allowed null values per row:",
        min_value=0,
        max_value=st.session_state.df.shape[1],
        value=3,
    )
    st.session_state.df = st.session_state.df[
        st.session_state.df.isnull().sum(axis=1) <= row_threshold
    ]
    st.write(f"✅ Removed rows with more than {row_threshold} null values.")

    # User-defined threshold for removing columns
    col_threshold = st.number_input(
        "Enter max allowed null values per column:",
        min_value=0,
        max_value=st.session_state.df.shape[0],
        value=100,
    )
    cols_to_drop = st.session_state.df.columns[
        st.session_state.df.isnull().sum() > col_threshold
    ].tolist()
    if cols_to_drop:
        st.session_state.df.drop(columns=cols_to_drop, inplace=True)
        st.write(f"✅ Dropped columns: {cols_to_drop}")
    else:
        st.write("✅ No columns removed.")

    # 4️⃣ Impute Missing Values
    st.subheader("📌 Impute Missing Values")
    for col in st.session_state.df.columns:
        if st.session_state.df[col].isnull().sum() > 0:
            method = st.selectbox(
                f"Choose method to fill missing values for {col}:",
                ["Mean", "Median", "Mode"],
            )
            if method == "Mean":
                st.session_state.df[col].fillna(
                    st.session_state.df[col].mean(), inplace=True
                )
            elif method == "Median":
                st.session_state.df[col].fillna(
                    st.session_state.df[col].median(), inplace=True
                )
            elif method == "Mode":
                st.session_state.df[col].fillna(
                    st.session_state.df[col].mode()[0], inplace=True
                )
            st.write(f"✅ {method} imputation applied for {col}")

    # Re-check null values after imputation
    st.write("**Final Null Values Count (After Imputation):**")
    st.write(st.session_state.df.isnull().sum())

    # 5️⃣ Drop Unimportant Columns
    st.subheader("🗑️ Drop Less Important Columns")
    selected_columns = st.multiselect(
        "Select columns to remove:", st.session_state.df.columns
    )
    if selected_columns:
        st.session_state.df.drop(columns=selected_columns, inplace=True)
        st.write(f"✅ Dropped columns: {selected_columns}")

    # 6️⃣ Target Variable Selection
    st.subheader("🎯 Target Variable Selection")
    target_col = st.selectbox("Select the target column:", st.session_state.df.columns)

    # 7️⃣ Problem Type Selection
    problem_type = st.radio(
        "Is this a Classification or Regression problem?",
        ("Classification", "Regression"),
    )

    # 8️⃣ Encoding Categorical Columns
    st.subheader("🔄 Encoding Categorical Features")
    categorical_cols = st.session_state.df.select_dtypes(
        include=["object"]
    ).columns.tolist()
    encoding_methods = {}

    if categorical_cols:
        for col in categorical_cols:
            encoding_methods[col] = st.selectbox(
                f"Choose encoding for {col}:", ["Label Encoding", "One-Hot Encoding"]
            )

        for col, method in encoding_methods.items():
            if method == "Label Encoding":
                le = LabelEncoder()
                st.session_state.df[col] = le.fit_transform(st.session_state.df[col])
            elif method == "One-Hot Encoding":
                st.session_state.df = pd.get_dummies(st.session_state.df, columns=[col])

        st.write("✅ Categorical encoding applied!")

    # 9️⃣ Normalization & Standardization
    st.subheader("⚙️ Feature Scaling")
    scaling_cols = st.multiselect(
        "Select columns to apply scaling:", st.session_state.df.columns
    )
    scaling_method = st.radio(
        "Choose scaling method:",
        ["Standardization (Z-score)", "Normalization (Min-Max)"],
    )

    if scaling_cols:
        scaler = (
            StandardScaler()
            if scaling_method.startswith("Standard")
            else MinMaxScaler()
        )
        st.session_state.df[scaling_cols] = scaler.fit_transform(
            st.session_state.df[scaling_cols]
        )
        st.write(f"✅ {scaling_method} applied on selected columns!")

    # 🔟 Train-Test Split
    st.subheader("📚 Train-Test Split")
    test_size = st.slider(
        "Select Test Data Percentage:",
        min_value=0.1,
        max_value=0.5,
        step=0.05,
        value=0.2,
    )

    X = st.session_state.df.drop(columns=[target_col])
    y = st.session_state.df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    st.write(f"✅ Training Data: {X_train.shape[0]} records")
    st.write(f"✅ Testing Data: {X_test.shape[0]} records")

    # 📊 Statistical Summary
    if st.button("Statistical Summary"):
        st.subheader("📊 Statistical Summary of Data")
        st.dataframe(st.session_state.df.describe())

    # 📥 Download Processed Data
    st.subheader("📥 Download Processed Data")
    file_format = st.radio("Select File Format:", ["CSV", "Excel"])

    def convert_df(current_df):
        if file_format == "CSV":
            return current_df.to_csv(index=False).encode("utf-8")
        elif file_format == "Excel":  # Corrected condition
            excel_buffer = io.BytesIO()  # ADDED: Create BytesIO buffer
            current_df.to_excel(
                excel_buffer, index=False, engine="openpyxl"
            )  # ADDED: Write to buffer with excel_writer
            return excel_buffer.getvalue()  # ADDED: Return bytes from buffer

    file_name = st.text_input("Enter file name:", "processed_data")

    if st.button("Download Data"):
        filedata = convert_df(
            st.session_state.df
        )  # call convert_df and store result to variable
        st.download_button(
            label="📥 Download Processed Data",
            data=filedata,  # use variable here
            file_name=f"{file_name}.{'csv' if file_format == 'CSV' else 'xlsx'}",  # corrected extension logic
            mime=(
                "text/csv"
                if file_format == "CSV"
                else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),  # corrected mime logic
        )

    return st.session_state.df, X_train, X_test, y_train, y_test
