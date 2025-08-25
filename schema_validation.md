## 📋 **Schema Validation System**

The application implements a robust JSON schema validation system to ensure consistent AI model outputs and reliable data processing:

### 🔒 **AI Model Output Schema**

- **Comprehensive Schema Definition**: Detailed JSON schema enforcing consistent structure for AI outputs
- **Type Validation**: Intelligent validation allowing both strict types and null values where appropriate
- **Required Fields**: Strict enforcement of critical fields while providing flexibility for optional data
- **Nested Structure**: Support for complex nested objects including strategy recommendations and market analysis
- **Field Documentation**: Descriptive comments for each field to ensure proper usage

### 🔄 **Data Adaptation Framework**

- **Structure Transformation**: Intelligent adaptation between flat and nested data structures
- **Default Values**: Automatic fallback to sensible defaults when fields are missing
- **Type Coercion**: Smart handling of type inconsistencies for numeric fields
- **Error Recovery**: Graceful handling of schema validation errors with informative messages
- **Flexible Processing**: Support for both strict and lenient validation depending on context

### 🛠️ **Implementation Details**

- **Schema Definition**: Central schema definition in `src/utils/ai_output_schema.py`
- **Validation Functions**: Comprehensive validation utilities with detailed error reporting
- **Integration Points**: Seamless integration with AI analysis pipeline and strategy processing
- **Adaptation Layer**: Intelligent transformation of non-conforming data into validated schema
- **Error Handling**: Graceful fallback mechanisms when validation fails

### 📊 **Benefits**

- **Reliability**: Eliminates unexpected structure or type errors in AI responses
- **Consistency**: Ensures uniform data structure across different application components
- **Documentation**: Self-documenting data contracts through schema definitions
- **Flexibility**: Supports adaptation of various data formats to expected schema
- **Error Prevention**: Catches data inconsistencies early in the processing pipeline
