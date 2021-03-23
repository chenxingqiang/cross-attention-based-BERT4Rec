#!/usr/bin/env python
# encoding: utf-8
"""
# @Time    : 2021/3/18 17:28
# @Author  : Chen.xingqiang
"""
# !/bin/python3
# coding:utf-8
# Copyright 2020 Alphaplato. All Rights Reserved.
# Desc:reading tfrecord

from __future__ import print_function

import base64
import requests
import json

# fill your ip and port
SERVER_URL = 'http://113.31.154.8:8051/v1/models/ybren_sdm:predict'


def main():
    single_sample = {"inputs": {"user_id": [[0]], "item_id": [[0]], "short_item_id": [[0]], "prefer_item_id": [
        [2830, 2351, 2827, 2353, 2118, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                "prefer_sess_length": [[5]], "short_sess_length": [[1]], "short_item_app_cate": [[371]],
                                "prefer_item_app_cate": [
                                    [540, 371, 540, 371, 631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0]], "user_gender": [[1]], "user_age": [[62]], "user_province": [[3]],
                                "user_city": [[26]], "user_is_vip": [[0]], "user_born_date": [[4220]],
                                "user_status": [[0]], "item_fabric_code": [[649]], "item_name": [[2533]],
                                "item_erp_sex": [[1]], "item_erpbig_cate": [[8]], "item_erpsmall_cate": [[4]],
                                "item_unit_price": [[19]], "item_shop_price": [[14]]}}

    many_samples = [single_sample] * 1000
    # predict_request = '{"instances" : [%s]}' % single_sample
    predict_request = '{"instances" :  %s}' % json.dumps(many_samples)
    # Send few requests to warm-up the model.
    for _ in range(1):
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()

    # Send few actual requests and report average latency.
    total_time = 0
    num_requests = 1000
    for _ in range(num_requests):
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()
        prediction = response.json()['predictions'][0]
    print('Prediction score: {}, avg latency: {} ms'.format(
        prediction[0], (total_time * 1000) / num_requests))


if __name__ == '__main__':
    main()
