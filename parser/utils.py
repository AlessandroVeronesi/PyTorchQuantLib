
## Parser Utils
def replace_module(model, old_module, replacer):
    # Recursively replace modules in the copied model
    for name, module in model.named_children():
        if not hasattr(module, 'nvdla'):
            if isinstance(module, old_module):
                # Instantiate the replacement module
                replacement = replacer(module, name)
                setattr(model, name, replacement)
            else:
                # If the module has children, apply recursively
                replace_module(module, old_module, replacer)

    return model  # Return the modified model copy

def replace_singleModule(model, module_name, replacer):
    # Recursively replace modules in the copied model
    for name, module in model.named_children():
        if hasattr(module, module_name):
            # Instantiate the replacement module
            replacement = replacer(module, name)
            setattr(model, name, replacement)
        else:
            # If the module has children, apply recursively
            replace_singleModule(module, module_name, replace_module)

    return model  # Return the modified model copy
