from wdb._engine import RustEnv

# Rust PyO3 classes are final by default and cannot be subclassed in Python.
# Since we don't need to add methods yet, we just create an alias.
CloudWDBEnv = RustEnv