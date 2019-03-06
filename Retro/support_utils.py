import smtplib
import numpy as np
import time
import re
import argparse
import os, glob

import matplotlib.pyplot as plt
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from os.path import basename



def visualize_experiment(path_to_experiment):
    exp_runs = glob.glob(f'{path_to_experiment}/**/monitor.csv', recursive=True)

    experiment_name = os.path.split(path_to_experiment)[1]

    all_runs = []
    max_reward = []
    last_rewards = []
    with open(os.path.join(path_to_experiment, f"README/README.txt"), "a+") as results_file:
        for index, run in enumerate(exp_runs):
            run_data = np.genfromtxt(run, delimiter=',')[1:]
            print(f"The {index}th run: {len(run_data)} episodes with a max reward of {np.max(np.array(run_data).transpose()[0]):.2f}.", file=results_file)
            all_runs.append(run_data)
        all_runs = np.array(all_runs)

        max_rewards = [np.max(np.array(run_data).transpose()[0]) for run_data in all_runs]
        last_rewards = [np.array(run_data).transpose()[0][-1] for run_data in all_runs]
        print(f"\nThe mean last results: {np.mean(last_rewards)} +- {np.std(last_rewards)}.", file=results_file)
        print(f"The mean max results: {np.mean(max_rewards)} +- {np.std(max_rewards)}.", file=results_file)

    fig, ax = plt.subplots()
    colors = ["#FFA07A", "#1E90FF"]
    ax.set(xlabel='Reset Count', ylabel='Reward', title=f'{experiment_name} combined results')
    max_len = max([len(run) for run in all_runs])
    all_runs_padded = []
    for run in all_runs:
        pad = np.empty((max_len - len(run), 3,))
        pad[:] = np.nan
        padded = np.concatenate((run,pad))
        all_runs_padded.append(padded)
    all_runs_padded = np.array(all_runs_padded)

    means = np.nanmean(all_runs_padded, axis=0)
    stds = np.nanstd(all_runs_padded, axis=0)
    mean_rewards = means.transpose()[0]
    std_rewards = stds.transpose()[0]

    reward = mean_rewards
    episodes = np.arange(len(mean_rewards))
    ax.plot(episodes, reward, color="#FFA07A")
    ax.fill_between(episodes, reward+std_rewards, reward-std_rewards, color="#FFA07A",  alpha=0.25)
    plt.savefig(os.path.join(path_to_experiment, f"README/{experiment_name}_cumulative.png"),  bbox_inches='tight')
    ax.grid()



    fig, ax = plt.subplots()
    for _, run in enumerate(all_runs):
        reward = run.transpose()[0]
        steps = np.arange(len(run.transpose()[1]))
        ax.plot(steps, reward, label=f"Run{_+1}")
    ax.set(xlabel='Reset Count', ylabel='Reward', title=f'{experiment_name} Seperate Results')
    ax.grid()
    ax.legend(loc='upper left', frameon=False)
    plt.savefig(os.path.join(path_to_experiment, f"README/{experiment_name}_seperate.png"),  bbox_inches='tight')


def visualize_run(path_to_logdir: str, timesteps=True):
    """
        path_to_logdir: The path to the run folder.
        timesteps: Whether to visualize in timestpes or episodes.
    """

    monitor_path = os.path.join(path_to_logdir, "monitor.csv")
    try:
        run_data = np.genfromtxt(monitor_path, delimiter=',')[1:]
        run_data = run_data.transpose()
    except OSError:
        print(f'\033[91mMonitor.csv not found at {path_to_logdir}. Continuing.')
        return

    #Set the names.
    experiment_path, run_name = os.path.split(path_to_logdir)
    experiment_name = os.path.basename(experiment_path)

    fig, ax = plt.subplots()
    xlabel = "Timesteps" if timesteps else "Episodes"
    ax.set(xlabel=xlabel, ylabel='Reward', title=f'{experiment_name}: {run_name}')

    # Mean Reward
    y_axis = run_data[0]

    # Timesteps or Episodes
    x_axis = np.cumsum(run_data[1]) if timesteps else np.arange(len(run_data[1]))
    ax.plot(x_axis, y_axis, color="#B22222", label="Mean Reward")
    ax.legend(loc='upper left', frameon=False)
    ax.grid()
    plt.savefig(os.path.join(path_to_logdir, f"mean_reward_{experiment_name}_{run_name}.png"))
    plt.savefig(os.path.join(experiment_path, f"README/mean_reward_{experiment_name}_{run_name}.png"))

def runningMean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
        y[ctr] = np.sum(x[ctr:(ctr + N)])
    return y / N


def visualize_cumulative_reward(input_file: str, ouput_destionation: str, readme_dest: str, experiment_name: str, run_count: int) -> None:
    indexes = []
    values = []
    N = 4

    with open(input_file, "r") as input:
        for line in input:
            indexes.append(int(line.split(",")[0]))
            value = float(line.split(",")[1])
            if value > 10000:
                value = values[-1]
            values.append(value)

    running_average = runningMean(values, N)

    fig = plt.figure()
    plt.title(f"Cumulative Reward for Run {run_count} of {experiment_name}")
    plt.plot(indexes, values, color="teal", linewidth=1.5, linestyle="-", label="Cumulative Reward")
    plt.plot(indexes, running_average, color="gray", linewidth=1, linestyle=":", label="Running Average of 4")
    plt.legend(loc='upper left', frameon=False)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    plt.savefig(ouput_destionation, bbox_inches='tight')
    plt.savefig(readme_dest, bbox_inches='tight')
    # plt.show()


def visualize_max_reward(input_file: str, ouput_destionation: str, readme_dest: str, experiment_name: str, run_count: int) -> None:
    indexes = []
    values = []
    N = 4

    with open(input_file, "r") as input:
        for line in input:
            indexes.append(int(line.split(",")[0]))
            value = float(line.split(",")[1])
            if value > 10000:
                value = values[-1]
            values.append(value)

    running_average = runningMean(values, N)

    fig = plt.figure()
    plt.title(f"Max Reward for Run {run_count} of {experiment_name}")
    plt.plot(indexes, values, color="orange", linewidth=1.5, linestyle="-", label="Max Reward")
    plt.plot(indexes, running_average, color="gray", linewidth=1, linestyle=":", label="Running Average of 4")
    plt.legend(loc='upper left', frameon=False)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    plt.savefig(ouput_destionation, bbox_inches='tight')
    plt.savefig(readme_dest, bbox_inches='tight')
    # plt.show()


def send_email(msg_body: str, run_path: str) -> None:
    """
    Sends the email to me!
    """
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("kafkabot9000@gmail.com", "thisisnotimportant")

    msg = MIMEMultipart()
    files = glob.glob(f'{run_path}/*.png')

    msg_body += "\n"
    msg_body += time.asctime(time.localtime(time.time()))
    msg.attach(MIMEText(msg_body))

    for f in files:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        msg.attach(part)

    server.sendmail("kafkabot9000@gmail.com", "baytemiz@ucsc.edu", msg.as_string())
    server.quit()

def parseArguments():
    parser = argparse.ArgumentParser()

    # parser.add_argument('-s', action='store', dest='save_prob', default=None,
    #                     help='The snapshot save probability [0,1]')
    #
    # parser.add_argument('-l', action='store', dest='load_prob',default=None,
    #                     help='The snapshot load probability [0,1]')
    #
    # parser.add_argument('-e', action='store', dest='experiment_name',default=None,
    #                     help='The name of the experiment')
    results = parser.parse_args()

    assert results.save_prob and results.load_prob and results.experiment_name, "\nPlease specify the save and load probablities with the flags -s and -l and the experiment name with the -e flag."
    print(f"\n\n\nTHE SAVE PROBABILITY: {results.save_prob}")
    print(f"THE LOAD PROBABILITY: {results.load_prob}\n\n\n")
    return (float(results.save_prob), float(results.load_prob), results.experiment_name)

def save_hyperparameters(filenames: list, path_to_file: str,  experiment_name:str, save_prob=None, load_prob=None, breadcrumb="# BREADCRUMBS") -> None:
    """
    Saves the lines in between breadcrumbs in all the given filenames. This is used for saving hyperparameters for RL training.
    Parameters
    ----------
    breadcrumb: Writes the lines between {breadcrumb}_START and {breadcrumb}_END.
    """
    with open(path_to_file, "a") as dest:
        dest.write(f"Experiment name: {experiment_name} \n")
        if save_prob:
            dest.write(f"Save prob:{save_prob} \n")
        if load_prob:
            dest.write(f"Load prob:{load_prob} \n")
        for filename in filenames:
            with open(filename, "r") as source:
                saving = False
                for line in source:
                    if line.strip() == f"{breadcrumb}_START":
                        dest.write("\n")
                        saving = True
                        continue
                    if line.strip() == f"{breadcrumb}_END":
                        saving = False
                        continue
                    if saving:
                        dest.write(line)
            print(f"{filename} hyperparameters have been saved!")
        print("Information saving is complete!")
