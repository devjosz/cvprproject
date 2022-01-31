import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas
import matplotlib.pyplot as plt

DEFAULT_FILENAME = "important_results.txt"


def savetofile(text, data, filename=DEFAULT_FILENAME):
    with open(filename, "a") as f:
        towrite = text + str(data) + "\n"
        f.write(towrite)


def eval_and_save(dataset,
                  model,
                  text,
                  filename=DEFAULT_FILENAME,
                  dictio=False):
    eval_res = model.evaluate(dataset, return_dict=dictio)
    savetofile(text, eval_res, filename)


def history_save(history_object, filename):
    """Saves the given history_object as a csv file withe the given filename"""
    datafr = pandas.DataFrame(history_object.history)
    datafr.to_csv(filename, index=False)


def history_plot(history_object, list_keys, text, filename, colors_style):
    """Plots the data contained in
    history_object.history[list_keys]. There need to be exactly two list
    keys. Text is a list containing precisely two strings, one will be the
    title of the x axis, the other the y axis
    """
    plt.figure()
    counter = 0
    for key in list_keys:
        plt.plot(history_object.history[key],colors_style[counter])
        counter = counter + 1
    plt.legend(list_keys)
    plt.xlabel(text[0])
    plt.ylabel(text[1])
    plt.savefig(filename)
