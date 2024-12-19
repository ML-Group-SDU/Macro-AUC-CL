
class SGDUpdate:
    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss_init = 0
            self.loss = 0

            def closure():
                # Forward
                self._before_forward(**kwargs)
                self.mb_output = self.forward()
                self._after_forward(**kwargs)

                # Loss & Backward
                self.loss = self.criterion()
                self._before_backward(**kwargs)
                self.loss.backward()
                self._after_backward(**kwargs)

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss = self.criterion()
            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)


            # Optimization step
            self._before_update(**kwargs)
            # self.optimizer_step()

            if self.use_closure:
                self.optimizer.step(
                    closure,
                )
            else:
                self.optimizer.step()

            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
