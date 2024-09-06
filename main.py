import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

data_path = "acquisition/"
df_list = []


def test():
    try:
        logging.debug("Starting test function")
        # Simulate some work
        result = "test"
        logging.debug("Finished test function")
        return result
    except Exception as e:
        logging.error(f"Error in test function: {e}")
        raise


with ProcessPoolExecutor() as executor:
    futures = [executor.submit(test) for i in range(10)]

    for i, future in enumerate(as_completed(futures)):
        try:
            logging.debug(f"Processing future {i}/{len(futures)}")
            result = future.result()
            print(f"{i}/{len(futures)}")
            print(result)
        except Exception as e:
            logging.error(f"Error processing future {i}: {e}")
