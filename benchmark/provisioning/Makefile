setup-vagrant:
	@vagrant up

configure-baremetal:
	ansible-playbook site.yml -i ./hosts -K

configure-baremetal-rootless:
	ansible-playbook site.yml -i ./hosts
