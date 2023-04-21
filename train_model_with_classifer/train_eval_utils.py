'''
train seq2seq+classifier
'''
import torch.nn.functional as F
import torch
import time
from tqdm import tqdm

def train_one_epoch(data_loader,eval_dataloader ,discriminator, optimizer,\
                epoch=1, log_interval=10, eval_step=30,device='cuda',file_path='xxx{}.pt'):
    samples_so_far = 0
    best_accuracy=0.8
    #记录样本数目
    #freeze LM and train classifier
    discriminator.train_custom()
    test_loss_l=[]
    test_accuracy_l=[]
    step_s=[]
    t1=time.perf_counter()
    for batch_idx,bath_data in enumerate(tqdm(data_loader)):
        input_t, target_t = bath_data['inputs'],bath_data['labels']
        input_t, target_t = input_t.to(device), target_t.to(device)
        optimizer.zero_grad()
        output_t = discriminator(input_t)
        # print(output_t.size())
        # print(target_t.size())
        loss = F.nll_loss(output_t, target_t)
        loss.backward(retain_graph=True)
        optimizer.step()
        samples_so_far += len(input_t)
        if batch_idx % log_interval == 0:
            t2=time.perf_counter()
            print(
                "Train Epoch(step): {}({}) [{}/{} ({:.0f}%),{:.4f}/s]\tLoss: {:.6f}".format(
                    epoch + 1,
                    batch_idx+1,
                    samples_so_far, len(data_loader.dataset),
                    100 * samples_so_far / len(data_loader.dataset), t2-t1,loss.item()
                )
            )
            t1=time.perf_counter()
        if (batch_idx %eval_step or batch_idx+1==len(data_loader)) and (batch_idx!=0)\
            and (loss.item()<0.12):
            test_loss, test_accuracy = evaluate_performance(
                data_loader=eval_dataloader,
                discriminator=discriminator,
                device=device
                )
            test_loss_l.append(test_loss)
            test_accuracy_l.append(test_accuracy)
            step_s.append(batch_idx)
            if test_accuracy>best_accuracy:
                torch.save(discriminator.get_classifier().state_dict(),
                        file_path.format('best'))
                best_accuracy=test_accuracy
    return test_loss_l,test_accuracy_l,step_s
                
            
def evaluate_performance(data_loader, discriminator, device='cuda'):
    discriminator.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for bath_data in data_loader:
            input_t, target_t = bath_data['inputs'],bath_data['labels']
            input_t, target_t = input_t.to(device), target_t.to(device)
            output_t = discriminator(input_t)
            # sum up batch loss
            test_loss += F.nll_loss(output_t, target_t, reduction="sum").item()
            # get the index of the max log-probability
            pred_t = output_t.argmax(dim=1, keepdim=True)
            correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()
    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    print(
        "Performance on test set: "
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(data_loader.dataset),
            100. * accuracy
        )
    )
    return test_loss, accuracy