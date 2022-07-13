from stable_baselines.common.callbacks import BaseCallback
# from stable_baselines3.common.callbacks import BaseCallback


class Original_Logger(BaseCallback):
    """
        tensorboardへの報酬グラフを追加する処理
        最適なモデルを log_dir に保存する処理
    """

    def __init__(self, gamma, verbose=2):
        super(Original_Logger, self).__init__(verbose)

        # 割引あり報酬の計算のための変数
        self.gamma, self.ep_len = gamma, 0
        self.episodeScore_discount = 0  # Score accumulated during an episode

        # 報酬関数の各項の和を計算するための変数
        self.details_reward = []    # 報酬和：空のリストで初期化(サイズは後に決定)
        self.details_reward_discount = []    # 割引報酬和：空のリストで初期化(サイズは後に決定)

    def get_attr_from_env(self, variable_name):
        return self.training_env.get_attr(variable_name)

    def _on_step(self):
        """
        tensorboardへの報酬グラフ(episode_reward + episode_lenght)を追加する処理
        """
        # terminal(終端状態)のときに self.locals['info'] に値が入る。 terminal以外は空のリスト : [{}]
        episode_dict = {}
        episode_dict = self.locals['infos'][0].get('episode')

        # 割引あり報酬の計算
        self.episodeScore_discount += (self.gamma ** (self.ep_len)) * self.locals['rewards'][0]

        if self.details_reward == []:   # リストのサイズを決定
            self.details_reward = [0] * len(self.get_attr_from_env('logging_reward')[0])

        if self.details_reward_discount == []:   # リストのサイズを決定
            self.details_reward_discount = [0] * len(self.get_attr_from_env('logging_reward')[0])

        # 環境のloggin_rewardから取得
        for i, val in enumerate(self.get_attr_from_env('logging_reward')[0]):
            self.details_reward[i] += val
            self.details_reward_discount[i] += (self.gamma ** (self.ep_len)) * val

        self.ep_len += 1

        if episode_dict != None:        # エピソードが終了したとき
            # episode_reward and episode_lenght add tensorboard
            episode_reward = episode_dict.get('r')      # 割引率をかけていない報酬(ΣR(s,a))
            episode_length = episode_dict.get('l')      # エピソードのステップ数

            # 終端状態におけるロボットの座標(環境から取得)
            terminal_robotPos = self.get_attr_from_env('logging_robotPos')[0][:3]

            """
            tensorboard でのログ保存処理
            """

            self.logger.record("original_logger/timeSteps",
                               self.num_timesteps)
            self.logger.record("original_logger/episode_lenght",
                               episode_length)
            self.logger.record("original_logger/episode_reward",
                               episode_reward)
            self.logger.record("original_logger/episode_discount_reward",
                               self.episodeScore_discount)
            self.logger.record("original_logger/robot_position_x",
                               terminal_robotPos[0])
            self.logger.record("original_logger/robot_position_y",
                               terminal_robotPos[2])
            self.logger.record("original_logger/robot_position_z",
                               terminal_robotPos[1])

            # 報酬の各項の和を保存
            for i, val in enumerate(self.details_reward):   # 報酬和
                loggin_name = "details_reward/" + str(i+1) + "-term"
                self.logger.record(loggin_name, val)

            for i, val in enumerate(self.details_reward_discount):   # 割引報酬和
                loggin_name = "details_reward/" + str(i+1) + "-term_discount"
                self.logger.record(loggin_name, val)

            self.logger.dump(self.num_timesteps)

            self.episodeScore_discount, self.ep_len = 0, 0
            self.details_reward = [0] * len(self.details_reward)
            self.details_reward_discount = [
                0] * len(self.details_reward_discount)

        return True
