import torch
import argparse
from DatasetFromCSV import *
from torch.autograd import Variable
import utils
from utils import *



parser = argparse.ArgumentParser(description='')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
args = parser.parse_args()

input_channels = 1
seq_length = int(7 / input_channels)
steps = 0
test_dataset = DatasetFromCSV("./test_fs.csv")

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

def test(model,max_accuracy, max_precise, max_recall, max_f1_score):
    confusion_matrix = torch.zeros(2, 2)
    accuracy = 0
    eval_loss = 0
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (parameters, lable) in enumerate(test_loader):

            parameters = utils.to_var(parameters)
            parameters = parameters.view(-1, input_channels, seq_length)
            output = model(parameters)
            lable = lable.type(torch.long)

            test_loss += F.nll_loss(output, lable, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(lable.data.view_as(pred)).cpu().sum()

            test_loss /= len(test_loader.dataset)

            pred = torch.squeeze(pred)

            for i in range(len(pred)):
                confusion_matrix[lable[i]][pred[i]] += 1


            precise, recall, f1_score = utils.CalConfusionMatrix(confusion_matrix)

            accuracy_value = correct.item() / len(test_loader.dataset)
            if accuracy_value > max_accuracy:
                max_accuracy = accuracy_value
            if precise > max_precise:
                max_precise = precise
            if recall > max_recall:
                max_recall = recall
            if f1_score > max_f1_score:
                max_f1_score = f1_score

            print('Evalution: Average loss: {:.4f}'.format(test_loss))
            print(
                'Current Evalution: Accuracy: {}/{} ({:.4f}), Precise: {:.4f}, ReCall: {:.4f}, F1 Score: {:.4f}'.format(
                    correct, len(test_loader.dataset), accuracy_value, precise, recall, f1_score))
            print(
                'Max Evalution: Accuracy: {:.4f}, Precise: {:.4f}, ReCall: {:.4f}, F1 Score: {:.4f}'.format(
                    max_accuracy, max_precise, max_recall, max_f1_score))

        return max_accuracy, max_precise, max_recall, max_f1_score,