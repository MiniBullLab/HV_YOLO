import os
import logger as Logger

def logger_test(save_path="./output"):
    logger = Logger(os.path.join(save_path, "logs"))

    # tensorboard logger
    infotrain = {
        'losstrain': loss.data
    }
    infoval = {
        'lossval': loss.data,
        'prec1': prec1,
        'prec5': prec5
    }

    if i % config.display == 0:
        if training:
            for tag, value in infotrain.items():
                logger.scalar_summary(tag, value, (epoch * (len(data_loader) - 1) + i))
        else:
            for tag, value in infoval.items():
                logger.scalar_summary(tag, value, (epoch * (len(data_loader) - 1) + i))

if __name__ == "__main__":
    logger_test()