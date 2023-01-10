def try_catch(func):
    ''' Decorator for try catch '''
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }, 400
    return wrapper
