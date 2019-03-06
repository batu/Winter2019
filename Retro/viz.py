def exp(path_to_experiment):
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


def run(path_to_logdir: str, timesteps=True):
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
